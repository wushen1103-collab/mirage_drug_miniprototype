from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold

from mirage_mini.data import (
    DatasetBundle,
    inject_missing_modalities,
    prepare_chembl_assay_subset,
    prepare_public_dti_official_splits,
    prepare_public_dti_subset,
    split_assay_cold,
    split_random,
    split_temporal,
    split_target_cold,
)
from mirage_mini.features import MorganFingerprintFeaturizer, MultiModalFeaturizer
from mirage_mini.metrics import evaluate_binary
from mirage_mini.model import (
    append_dense_features,
    build_reliability_features,
    blend_probabilities,
    build_gate_features,
    fit_arbitration_gate,
    fit_gate_model,
    fit_reliability_probe,
    select_best_candidate,
    select_blend_weight,
    train_classifier,
)
from mirage_mini.neural_probe import predict_interaction_probe, train_interaction_probe
from mirage_mini.retrieval import MultiViewRetrievalAugmentor, RetrievalAugmentor, TanimotoRetrievalAugmentor

ROBUST_SEQUENCE_ANCHOR_ALPHA = 0.35
ROBUST_PROBE_ANCHOR_ALPHA = 0.45
RETRIEVAL_SEQUENCE_BRIDGE_ALPHA = 0.45


def _fit_conflict_aware_arbitration(
    *,
    y_train: np.ndarray,
    y_val: np.ndarray,
    current_val: np.ndarray,
    retrieval_val: np.ndarray,
    current_test: np.ndarray,
    retrieval_test: np.ndarray,
    current_missing: np.ndarray,
    retrieval_missing: np.ndarray,
    val_stats: np.ndarray,
    test_stats: np.ndarray,
    missing_stats: np.ndarray,
    val_masks: np.ndarray,
    test_masks: np.ndarray,
    missing_masks: np.ndarray,
    seed: int = 42,
) -> dict[str, np.ndarray | float]:
    """Cross-fit gate decisions before training the reliability probe on them."""
    y_val = np.asarray(y_val, dtype=int)
    gate_features_val = build_gate_features(current_val, retrieval_val, val_stats, val_masks)
    better_current = ((np.asarray(current_val) - y_val) ** 2 <= (np.asarray(retrieval_val) - y_val) ** 2).astype(int)
    gamma_oof = np.full(len(y_val), 0.5, dtype=np.float32)
    _, counts = np.unique(y_val, return_counts=True)
    folds = min(3, int(counts.min())) if len(counts) == 2 else 0
    if folds >= 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for train_idx, holdout_idx in splitter.split(gate_features_val, y_val):
            fold_gate = fit_arbitration_gate(gate_features_val[train_idx], better_current[train_idx])
            gamma_oof[holdout_idx] = fold_gate.predict_weight(gate_features_val[holdout_idx])
    gated_oof = gamma_oof * np.asarray(current_val) + (1.0 - gamma_oof) * np.asarray(retrieval_val)
    reliability_features_oof = build_reliability_features(
        current_val,
        retrieval_val,
        gated_oof,
        gamma_oof,
        val_stats,
        val_masks,
    )
    correctness = ((gated_oof >= 0.5).astype(int) == y_val).astype(int)
    reliability_oof = np.full(len(y_val), float(np.mean(correctness)), dtype=np.float32)
    if folds >= 2:
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + 1)
        for train_idx, holdout_idx in splitter.split(reliability_features_oof, y_val):
            fold_probe = fit_reliability_probe(
                reliability_features_oof[train_idx], correctness[train_idx]
            )
            reliability_oof[holdout_idx] = fold_probe.predict_reliability(
                reliability_features_oof[holdout_idx]
            )
    reliability_probe = fit_reliability_probe(reliability_features_oof, correctness)
    gate = fit_arbitration_gate(gate_features_val, better_current)
    anchor = float(np.mean(np.asarray(y_train, dtype=float)))

    # The anchor is a validation-calibrated shrinkage target, not an unconditional
    # replacement for branch evidence. Cross-fitted validation predictions choose its
    # strength before the held-out test split is evaluated.
    anchor_strength_grid = np.linspace(0.0, 1.0, 21)
    anchor_losses = []
    for strength in anchor_strength_grid:
        anchor_mix = (1.0 - strength) * gated_oof + strength * anchor
        anchored_oof = reliability_oof * gated_oof + (1.0 - reliability_oof) * anchor_mix
        anchor_losses.append(float(np.mean((anchored_oof - y_val) ** 2)))
    anchor_strength = float(anchor_strength_grid[int(np.argmin(anchor_losses))])

    def apply(
        current: np.ndarray,
        retrieval: np.ndarray,
        stats: np.ndarray,
        masks: np.ndarray,
        gamma: np.ndarray | None = None,
        reliability_override: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        if gamma is None:
            features = build_gate_features(current, retrieval, stats, masks)
            gamma = gate.predict_weight(features)
        gated = gamma * np.asarray(current) + (1.0 - gamma) * np.asarray(retrieval)
        reliability_features = build_reliability_features(current, retrieval, gated, gamma, stats, masks)
        reliability = (
            np.asarray(reliability_override, dtype=np.float32)
            if reliability_override is not None
            else reliability_probe.predict_reliability(reliability_features)
        )
        equal = 0.5 * np.asarray(current) + 0.5 * np.asarray(retrieval)
        anchor_mix = (1.0 - anchor_strength) * gated + anchor_strength * anchor
        equal_anchor_mix = (1.0 - anchor_strength) * equal + anchor_strength * anchor
        no_gate = reliability * equal + (1.0 - reliability) * equal_anchor_mix
        no_probe = 0.5 * gated + 0.5 * anchor_mix
        no_anchor = gated
        full = reliability * gated + (1.0 - reliability) * anchor_mix
        return {
            "full": np.asarray(full, dtype=np.float32),
            "no_gate": np.asarray(no_gate, dtype=np.float32),
            "no_probe": np.asarray(no_probe, dtype=np.float32),
            "no_anchor": np.asarray(no_anchor, dtype=np.float32),
            "gamma": np.asarray(gamma, dtype=np.float32),
            "reliability": np.asarray(reliability, dtype=np.float32),
            "gated": np.asarray(gated, dtype=np.float32),
        }

    val_output = apply(
        current_val,
        retrieval_val,
        val_stats,
        val_masks,
        gamma=gamma_oof,
        reliability_override=reliability_oof,
    )
    test_output = apply(current_test, retrieval_test, test_stats, test_masks)
    missing_output = apply(current_missing, retrieval_missing, missing_stats, missing_masks)
    return {
        "val": val_output,
        "test": test_output,
        "missing": missing_output,
        "anchor": anchor,
        "anchor_strength": anchor_strength,
        "probe_oof_accuracy": float(np.mean(correctness)),
    }


def load_bundle(dataset: str, cache_dir: Path, sample_size: int, seed: int) -> DatasetBundle:
    if dataset.upper() == "CHEMBL_ASSAY":
        return prepare_chembl_assay_subset(
            cache_dir=cache_dir,
            sample_size=sample_size,
            seed=seed,
        )
    return prepare_public_dti_subset(
        dataset_name=dataset,
        cache_dir=cache_dir,
        sample_size=sample_size,
        seed=seed,
    )


def make_splits(df: pd.DataFrame, split_mode: str, seed: int) -> Dict[str, pd.DataFrame]:
    split_mode = split_mode.lower()
    if split_mode == "target_cold":
        return split_target_cold(df, seed=seed)
    if split_mode == "assay_cold":
        return split_assay_cold(df, seed=seed)
    if split_mode == "temporal":
        return split_temporal(df, seed=seed)
    if split_mode == "random":
        return split_random(df, seed=seed)
    raise ValueError(f"Unsupported split_mode: {split_mode}")


def run_model_suite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stressed_test_df: pd.DataFrame,
    n_neighbors: int = 8,
    smiles_embedder=None,
    text_embedder=None,
    sequence_embedder=None,
    enable_interaction_probe: bool = False,
    interaction_probe_config: dict | None = None,
    retrieval_reference_size: int | None = None,
) -> dict:
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    no_mask_feat = MultiModalFeaturizer(include_masks=False)
    mask_feat = MultiModalFeaturizer(include_masks=True)
    morgan_feat = MorganFingerprintFeaturizer(n_bits=2048, radius=2)

    x_train_nomask = no_mask_feat.transform(train_df)
    x_val_nomask = no_mask_feat.transform(val_df)
    x_test_nomask = no_mask_feat.transform(test_df)
    x_test_missing_nomask = no_mask_feat.transform(stressed_test_df)

    no_mask_model = train_classifier(x_train_nomask.matrix, y_train)
    no_mask_val_prob = no_mask_model.predict_proba(x_val_nomask.matrix)[:, 1]
    no_mask_test_prob = no_mask_model.predict_proba(x_test_nomask.matrix)[:, 1]
    no_mask_missing_prob = no_mask_model.predict_proba(x_test_missing_nomask.matrix)[:, 1]

    x_train_mask = mask_feat.transform(train_df)
    x_val_mask = mask_feat.transform(val_df)
    x_test_mask = mask_feat.transform(test_df)
    x_test_missing_mask = mask_feat.transform(stressed_test_df)

    mask_model = train_classifier(x_train_mask.matrix, y_train)
    mask_val_prob = mask_model.predict_proba(x_val_mask.matrix)[:, 1]
    mask_test_prob = mask_model.predict_proba(x_test_mask.matrix)[:, 1]
    mask_missing_prob = mask_model.predict_proba(x_test_missing_mask.matrix)[:, 1]

    retrieval = RetrievalAugmentor(
        n_neighbors=n_neighbors,
        max_reference_size=retrieval_reference_size,
    ).fit(x_train_mask.matrix, y_train)
    x_train_aug = append_dense_features(x_train_mask.matrix, retrieval.transform_train(x_train_mask.matrix))
    x_val_aug = append_dense_features(x_val_mask.matrix, retrieval.transform(x_val_mask.matrix))
    x_test_aug = append_dense_features(x_test_mask.matrix, retrieval.transform(x_test_mask.matrix))
    x_test_missing_aug = append_dense_features(
        x_test_missing_mask.matrix,
        retrieval.transform(x_test_missing_mask.matrix),
    )

    retrieval_model = train_classifier(x_train_aug, y_train)
    retrieval_val_prob = retrieval_model.predict_proba(x_val_aug)[:, 1]
    retrieval_test_prob = retrieval_model.predict_proba(x_test_aug)[:, 1]
    retrieval_missing_prob = retrieval_model.predict_proba(x_test_missing_aug)[:, 1]

    multiview_retrieval = MultiViewRetrievalAugmentor(
        n_neighbors=n_neighbors,
        max_reference_size=retrieval_reference_size,
    ).fit(x_train_mask, y_train)
    train_mv_stats = multiview_retrieval.transform_train(x_train_mask)
    val_mv_stats = multiview_retrieval.transform(x_val_mask)
    test_mv_stats = multiview_retrieval.transform(x_test_mask)
    missing_mv_stats = multiview_retrieval.transform(x_test_missing_mask)
    smiles_idx = multiview_retrieval.view_names.index("smiles")
    smiles_slice = slice(
        smiles_idx * multiview_retrieval.stats_dim,
        (smiles_idx + 1) * multiview_retrieval.stats_dim,
    )

    x_train_mv = append_dense_features(x_train_mask.matrix, train_mv_stats)
    x_val_mv = append_dense_features(x_val_mask.matrix, val_mv_stats)
    x_test_mv = append_dense_features(x_test_mask.matrix, test_mv_stats)
    x_test_missing_mv = append_dense_features(x_test_missing_mask.matrix, missing_mv_stats)

    multiview_model = train_classifier(x_train_mv, y_train)
    multiview_val_prob = multiview_model.predict_proba(x_val_mv)[:, 1]
    multiview_test_prob = multiview_model.predict_proba(x_test_mv)[:, 1]
    multiview_missing_prob = multiview_model.predict_proba(x_test_missing_mv)[:, 1]

    # This branch uses only statistics of training-memory neighbors. Keeping it
    # separate from the current-feature classifier makes arbitration auditable.
    retrieval_evidence_model = train_classifier(sparse.csr_matrix(train_mv_stats), y_train)
    retrieval_evidence_val_prob = retrieval_evidence_model.predict_proba(sparse.csr_matrix(val_mv_stats))[:, 1]
    retrieval_evidence_test_prob = retrieval_evidence_model.predict_proba(sparse.csr_matrix(test_mv_stats))[:, 1]
    retrieval_evidence_missing_prob = retrieval_evidence_model.predict_proba(sparse.csr_matrix(missing_mv_stats))[:, 1]

    x_train_smiles = append_dense_features(x_train_mask.matrix, train_mv_stats[:, smiles_slice])
    x_val_smiles = append_dense_features(x_val_mask.matrix, val_mv_stats[:, smiles_slice])
    x_test_smiles = append_dense_features(x_test_mask.matrix, test_mv_stats[:, smiles_slice])
    x_test_missing_smiles = append_dense_features(x_test_missing_mask.matrix, missing_mv_stats[:, smiles_slice])

    smiles_retrieval_model = train_classifier(x_train_smiles, y_train)
    smiles_retrieval_val_prob = smiles_retrieval_model.predict_proba(x_val_smiles)[:, 1]
    smiles_retrieval_test_prob = smiles_retrieval_model.predict_proba(x_test_smiles)[:, 1]
    smiles_retrieval_missing_prob = smiles_retrieval_model.predict_proba(x_test_missing_smiles)[:, 1]

    x_train_morgan = morgan_feat.transform(train_df["smiles"])
    x_val_morgan = morgan_feat.transform(val_df["smiles"])
    x_test_morgan = morgan_feat.transform(test_df["smiles"])
    x_test_missing_morgan = morgan_feat.transform(stressed_test_df["smiles"])

    morgan_retrieval = TanimotoRetrievalAugmentor(
        n_neighbors=n_neighbors,
        max_reference_size=retrieval_reference_size,
    ).fit(x_train_morgan, y_train)
    train_morgan_stats = morgan_retrieval.transform_train(x_train_morgan)
    val_morgan_stats = morgan_retrieval.transform(x_val_morgan)
    test_morgan_stats = morgan_retrieval.transform(x_test_morgan)
    missing_morgan_stats = morgan_retrieval.transform(x_test_missing_morgan)

    x_train_morgan_aug = append_dense_features(x_train_mask.matrix, train_morgan_stats)
    x_val_morgan_aug = append_dense_features(x_val_mask.matrix, val_morgan_stats)
    x_test_morgan_aug = append_dense_features(x_test_mask.matrix, test_morgan_stats)
    x_test_missing_morgan_aug = append_dense_features(x_test_missing_mask.matrix, missing_morgan_stats)

    morgan_smiles_model = train_classifier(x_train_morgan_aug, y_train)
    morgan_smiles_val_prob = morgan_smiles_model.predict_proba(x_val_morgan_aug)[:, 1]
    morgan_smiles_test_prob = morgan_smiles_model.predict_proba(x_test_morgan_aug)[:, 1]
    morgan_smiles_missing_prob = morgan_smiles_model.predict_proba(x_test_missing_morgan_aug)[:, 1]

    x_train_hybrid = append_dense_features(x_train_smiles, train_morgan_stats)
    x_val_hybrid = append_dense_features(x_val_smiles, val_morgan_stats)
    x_test_hybrid = append_dense_features(x_test_smiles, test_morgan_stats)
    x_test_missing_hybrid = append_dense_features(x_test_missing_smiles, missing_morgan_stats)

    hybrid_smiles_model = train_classifier(x_train_hybrid, y_train)
    hybrid_smiles_val_prob = hybrid_smiles_model.predict_proba(x_val_hybrid)[:, 1]
    hybrid_smiles_test_prob = hybrid_smiles_model.predict_proba(x_test_hybrid)[:, 1]
    hybrid_smiles_missing_prob = hybrid_smiles_model.predict_proba(x_test_missing_hybrid)[:, 1]

    x_train_hybrid_plus = sparse.hstack([x_train_hybrid, x_train_morgan], format="csr")
    x_val_hybrid_plus = sparse.hstack([x_val_hybrid, x_val_morgan], format="csr")
    x_test_hybrid_plus = sparse.hstack([x_test_hybrid, x_test_morgan], format="csr")
    x_test_missing_hybrid_plus = sparse.hstack([x_test_missing_hybrid, x_test_missing_morgan], format="csr")

    hybrid_plus_model = train_classifier(x_train_hybrid_plus, y_train)
    hybrid_plus_val_prob = hybrid_plus_model.predict_proba(x_val_hybrid_plus)[:, 1]
    hybrid_plus_test_prob = hybrid_plus_model.predict_proba(x_test_hybrid_plus)[:, 1]
    hybrid_plus_missing_prob = hybrid_plus_model.predict_proba(x_test_missing_hybrid_plus)[:, 1]

    hybrid_blend_val_prob = 0.5 * (hybrid_smiles_val_prob + hybrid_plus_val_prob)
    hybrid_blend_test_prob = 0.5 * (hybrid_smiles_test_prob + hybrid_plus_test_prob)
    hybrid_blend_missing_prob = 0.5 * (hybrid_smiles_missing_prob + hybrid_plus_missing_prob)

    pretrained_dense_val_prob = None
    pretrained_dense_test_prob = None
    pretrained_dense_missing_prob = None
    pretrained_retrieval_val_prob = None
    pretrained_retrieval_test_prob = None
    pretrained_retrieval_missing_prob = None
    hybrid_pretrained_val_prob = None
    hybrid_pretrained_test_prob = None
    hybrid_pretrained_missing_prob = None
    hybrid_pretrained_blend_alpha = None
    hybrid_pretrained_blend_val_prob = None
    hybrid_pretrained_blend_test_prob = None
    hybrid_pretrained_blend_missing_prob = None
    hybrid_pretrained_selector_name = None
    hybrid_pretrained_select_val_prob = None
    hybrid_pretrained_select_test_prob = None
    hybrid_pretrained_select_missing_prob = None
    hybrid_text_val_prob = None
    hybrid_text_test_prob = None
    hybrid_text_missing_prob = None
    hybrid_smiles_text_val_prob = None
    hybrid_smiles_text_test_prob = None
    hybrid_smiles_text_missing_prob = None
    pretrained_text_retrieval_val_prob = None
    pretrained_text_retrieval_test_prob = None
    pretrained_text_retrieval_missing_prob = None
    hybrid_text_retrieval_val_prob = None
    hybrid_text_retrieval_test_prob = None
    hybrid_text_retrieval_missing_prob = None
    hybrid_smiles_text_retrieval_val_prob = None
    hybrid_smiles_text_retrieval_test_prob = None
    hybrid_smiles_text_retrieval_missing_prob = None
    hybrid_text_blend_alpha = None
    hybrid_text_blend_val_prob = None
    hybrid_text_blend_test_prob = None
    hybrid_text_blend_missing_prob = None
    hybrid_smiles_text_blend_alpha = None
    hybrid_smiles_text_blend_val_prob = None
    hybrid_smiles_text_blend_test_prob = None
    hybrid_smiles_text_blend_missing_prob = None
    hybrid_sequence_val_prob = None
    hybrid_sequence_test_prob = None
    hybrid_sequence_missing_prob = None
    hybrid_smiles_sequence_val_prob = None
    hybrid_smiles_sequence_test_prob = None
    hybrid_smiles_sequence_missing_prob = None
    strong_gate_val_prob = None
    strong_gate_test_prob = None
    strong_gate_missing_prob = None
    gated_strong_pretrained_blend_alpha = None
    gated_strong_pretrained_val_prob = None
    gated_strong_pretrained_test_prob = None
    gated_strong_pretrained_missing_prob = None
    robust_sequence_anchor_alpha = None
    robust_sequence_anchor_val_prob = None
    robust_sequence_anchor_test_prob = None
    robust_sequence_anchor_missing_prob = None
    robust_probe_anchor_alpha = None
    robust_probe_anchor_val_prob = None
    robust_probe_anchor_test_prob = None
    robust_probe_anchor_missing_prob = None
    retrieval_sequence_bridge_alpha = None
    retrieval_sequence_bridge_val_prob = None
    retrieval_sequence_bridge_test_prob = None
    retrieval_sequence_bridge_missing_prob = None
    sequence_gate_val_prob = None
    sequence_gate_test_prob = None
    sequence_gate_missing_prob = None
    interaction_gate_blend_alpha = None
    interaction_gate_val_prob = None
    interaction_gate_test_prob = None
    interaction_gate_missing_prob = None
    fullsuite_val_select_name = None
    fullsuite_val_select_val_prob = None
    fullsuite_val_select_test_prob = None
    fullsuite_val_select_missing_prob = None
    interaction_probe_val_prob = None
    interaction_probe_test_prob = None
    interaction_probe_missing_prob = None
    interaction_probe_blend_alpha = None
    interaction_probe_blend_val_prob = None
    interaction_probe_blend_test_prob = None
    interaction_probe_blend_missing_prob = None
    interaction_probe_best_epoch = None
    interaction_probe_best_val_auprc = None
    interaction_probe_selected_baseline = None
    interaction_probe_text_field = None
    mirage_full_val_prob = None
    mirage_full_test_prob = None
    mirage_full_missing_prob = None
    mirage_no_gate_val_prob = None
    mirage_no_gate_test_prob = None
    mirage_no_gate_missing_prob = None
    mirage_no_probe_val_prob = None
    mirage_no_probe_test_prob = None
    mirage_no_probe_missing_prob = None
    mirage_no_anchor_val_prob = None
    mirage_no_anchor_test_prob = None
    mirage_no_anchor_missing_prob = None
    mirage_gate_val = None
    mirage_gate_test = None
    mirage_gate_missing = None
    mirage_reliability_val = None
    mirage_reliability_test = None
    mirage_reliability_missing = None
    mirage_anchor_probability = None
    mirage_anchor_strength = None
    mirage_probe_oof_accuracy = None

    if smiles_embedder is not None:
        train_smiles_emb = smiles_embedder.transform(train_df["smiles"])
        val_smiles_emb = smiles_embedder.transform(val_df["smiles"])
        test_smiles_emb = smiles_embedder.transform(test_df["smiles"])
        missing_smiles_emb = smiles_embedder.transform(stressed_test_df["smiles"])

        x_train_pretrained_dense = append_dense_features(x_train_mask.matrix, train_smiles_emb)
        x_val_pretrained_dense = append_dense_features(x_val_mask.matrix, val_smiles_emb)
        x_test_pretrained_dense = append_dense_features(x_test_mask.matrix, test_smiles_emb)
        x_missing_pretrained_dense = append_dense_features(x_test_missing_mask.matrix, missing_smiles_emb)

        pretrained_dense_model = train_classifier(x_train_pretrained_dense, y_train)
        pretrained_dense_val_prob = pretrained_dense_model.predict_proba(x_val_pretrained_dense)[:, 1]
        pretrained_dense_test_prob = pretrained_dense_model.predict_proba(x_test_pretrained_dense)[:, 1]
        pretrained_dense_missing_prob = pretrained_dense_model.predict_proba(x_missing_pretrained_dense)[:, 1]

        train_smiles_emb_sparse = sparse.csr_matrix(train_smiles_emb)
        val_smiles_emb_sparse = sparse.csr_matrix(val_smiles_emb)
        test_smiles_emb_sparse = sparse.csr_matrix(test_smiles_emb)
        missing_smiles_emb_sparse = sparse.csr_matrix(missing_smiles_emb)
        pretrained_retrieval = RetrievalAugmentor(
            n_neighbors=n_neighbors,
            max_reference_size=retrieval_reference_size,
        ).fit(train_smiles_emb_sparse, y_train)
        train_pretrained_stats = pretrained_retrieval.transform_train(train_smiles_emb_sparse)
        val_pretrained_stats = pretrained_retrieval.transform(val_smiles_emb_sparse)
        test_pretrained_stats = pretrained_retrieval.transform(test_smiles_emb_sparse)
        missing_pretrained_stats = pretrained_retrieval.transform(missing_smiles_emb_sparse)

        x_train_pretrained_retrieval = append_dense_features(x_train_pretrained_dense, train_pretrained_stats)
        x_val_pretrained_retrieval = append_dense_features(x_val_pretrained_dense, val_pretrained_stats)
        x_test_pretrained_retrieval = append_dense_features(x_test_pretrained_dense, test_pretrained_stats)
        x_missing_pretrained_retrieval = append_dense_features(x_missing_pretrained_dense, missing_pretrained_stats)

        pretrained_retrieval_model = train_classifier(x_train_pretrained_retrieval, y_train)
        pretrained_retrieval_val_prob = pretrained_retrieval_model.predict_proba(x_val_pretrained_retrieval)[:, 1]
        pretrained_retrieval_test_prob = pretrained_retrieval_model.predict_proba(x_test_pretrained_retrieval)[:, 1]
        pretrained_retrieval_missing_prob = pretrained_retrieval_model.predict_proba(x_missing_pretrained_retrieval)[:, 1]

        x_train_hybrid_pretrained = append_dense_features(x_train_hybrid_plus, train_smiles_emb)
        x_val_hybrid_pretrained = append_dense_features(x_val_hybrid_plus, val_smiles_emb)
        x_test_hybrid_pretrained = append_dense_features(x_test_hybrid_plus, test_smiles_emb)
        x_missing_hybrid_pretrained = append_dense_features(x_test_missing_hybrid_plus, missing_smiles_emb)

        hybrid_pretrained_model = train_classifier(x_train_hybrid_pretrained, y_train)
        hybrid_pretrained_val_prob = hybrid_pretrained_model.predict_proba(x_val_hybrid_pretrained)[:, 1]
        hybrid_pretrained_test_prob = hybrid_pretrained_model.predict_proba(x_test_hybrid_pretrained)[:, 1]
        hybrid_pretrained_missing_prob = hybrid_pretrained_model.predict_proba(x_missing_hybrid_pretrained)[:, 1]

        hybrid_pretrained_blend_alpha = select_blend_weight(
            y_val,
            hybrid_blend_val_prob,
            hybrid_pretrained_val_prob,
        )
        hybrid_pretrained_blend_val_prob = blend_probabilities(
            hybrid_blend_val_prob,
            hybrid_pretrained_val_prob,
            hybrid_pretrained_blend_alpha,
        )
        hybrid_pretrained_blend_test_prob = blend_probabilities(
            hybrid_blend_test_prob,
            hybrid_pretrained_test_prob,
            hybrid_pretrained_blend_alpha,
        )
        hybrid_pretrained_blend_missing_prob = blend_probabilities(
            hybrid_blend_missing_prob,
            hybrid_pretrained_missing_prob,
            hybrid_pretrained_blend_alpha,
        )

        hybrid_pretrained_selector_name = select_best_candidate(
            y_val,
            {
                "hybrid_blend_avg": hybrid_blend_val_prob,
                "hybrid_plus_pretrained_smiles": hybrid_pretrained_val_prob,
            },
        )
        selected_probs = {
            "hybrid_blend_avg": (
                hybrid_blend_val_prob,
                hybrid_blend_test_prob,
                hybrid_blend_missing_prob,
            ),
            "hybrid_plus_pretrained_smiles": (
                hybrid_pretrained_val_prob,
                hybrid_pretrained_test_prob,
                hybrid_pretrained_missing_prob,
            ),
        }
        (
            hybrid_pretrained_select_val_prob,
            hybrid_pretrained_select_test_prob,
            hybrid_pretrained_select_missing_prob,
        ) = selected_probs[hybrid_pretrained_selector_name]

    if sequence_embedder is not None:
        train_sequence_emb = sequence_embedder.transform(train_df["sequence"])
        val_sequence_emb = sequence_embedder.transform(val_df["sequence"])
        test_sequence_emb = sequence_embedder.transform(test_df["sequence"])
        missing_sequence_emb = sequence_embedder.transform(stressed_test_df["sequence"])

        x_train_hybrid_sequence = append_dense_features(x_train_hybrid_plus, train_sequence_emb)
        x_val_hybrid_sequence = append_dense_features(x_val_hybrid_plus, val_sequence_emb)
        x_test_hybrid_sequence = append_dense_features(x_test_hybrid_plus, test_sequence_emb)
        x_missing_hybrid_sequence = append_dense_features(x_test_missing_hybrid_plus, missing_sequence_emb)

        hybrid_sequence_model = train_classifier(x_train_hybrid_sequence, y_train)
        hybrid_sequence_val_prob = hybrid_sequence_model.predict_proba(x_val_hybrid_sequence)[:, 1]
        hybrid_sequence_test_prob = hybrid_sequence_model.predict_proba(x_test_hybrid_sequence)[:, 1]
        hybrid_sequence_missing_prob = hybrid_sequence_model.predict_proba(x_missing_hybrid_sequence)[:, 1]

        if smiles_embedder is not None:
            x_train_hybrid_smiles_sequence = append_dense_features(x_train_hybrid_pretrained, train_sequence_emb)
            x_val_hybrid_smiles_sequence = append_dense_features(x_val_hybrid_pretrained, val_sequence_emb)
            x_test_hybrid_smiles_sequence = append_dense_features(x_test_hybrid_pretrained, test_sequence_emb)
            x_missing_hybrid_smiles_sequence = append_dense_features(x_missing_hybrid_pretrained, missing_sequence_emb)

            hybrid_smiles_sequence_model = train_classifier(x_train_hybrid_smiles_sequence, y_train)
            hybrid_smiles_sequence_val_prob = hybrid_smiles_sequence_model.predict_proba(x_val_hybrid_smiles_sequence)[:, 1]
            hybrid_smiles_sequence_test_prob = hybrid_smiles_sequence_model.predict_proba(x_test_hybrid_smiles_sequence)[:, 1]
            hybrid_smiles_sequence_missing_prob = hybrid_smiles_sequence_model.predict_proba(x_missing_hybrid_smiles_sequence)[:, 1]

    if text_embedder is not None:
        train_text_emb = text_embedder.transform(train_df["text"])
        val_text_emb = text_embedder.transform(val_df["text"])
        test_text_emb = text_embedder.transform(test_df["text"])
        missing_text_emb = text_embedder.transform(stressed_test_df["text"])

        train_text_emb_sparse = sparse.csr_matrix(train_text_emb)
        val_text_emb_sparse = sparse.csr_matrix(val_text_emb)
        test_text_emb_sparse = sparse.csr_matrix(test_text_emb)
        missing_text_emb_sparse = sparse.csr_matrix(missing_text_emb)
        text_retrieval = RetrievalAugmentor(
            n_neighbors=n_neighbors,
            max_reference_size=retrieval_reference_size,
        ).fit(train_text_emb_sparse, y_train)
        train_text_stats = text_retrieval.transform_train(train_text_emb_sparse)
        val_text_stats = text_retrieval.transform(val_text_emb_sparse)
        test_text_stats = text_retrieval.transform(test_text_emb_sparse)
        missing_text_stats = text_retrieval.transform(missing_text_emb_sparse)

        x_train_text_retrieval = append_dense_features(x_train_mask.matrix, train_text_stats)
        x_val_text_retrieval = append_dense_features(x_val_mask.matrix, val_text_stats)
        x_test_text_retrieval = append_dense_features(x_test_mask.matrix, test_text_stats)
        x_missing_text_retrieval = append_dense_features(x_test_missing_mask.matrix, missing_text_stats)

        pretrained_text_retrieval_model = train_classifier(x_train_text_retrieval, y_train)
        pretrained_text_retrieval_val_prob = pretrained_text_retrieval_model.predict_proba(x_val_text_retrieval)[:, 1]
        pretrained_text_retrieval_test_prob = pretrained_text_retrieval_model.predict_proba(x_test_text_retrieval)[:, 1]
        pretrained_text_retrieval_missing_prob = pretrained_text_retrieval_model.predict_proba(x_missing_text_retrieval)[:, 1]

        x_train_hybrid_text = append_dense_features(x_train_hybrid_plus, train_text_emb)
        x_val_hybrid_text = append_dense_features(x_val_hybrid_plus, val_text_emb)
        x_test_hybrid_text = append_dense_features(x_test_hybrid_plus, test_text_emb)
        x_missing_hybrid_text = append_dense_features(x_test_missing_hybrid_plus, missing_text_emb)

        hybrid_text_model = train_classifier(x_train_hybrid_text, y_train)
        hybrid_text_val_prob = hybrid_text_model.predict_proba(x_val_hybrid_text)[:, 1]
        hybrid_text_test_prob = hybrid_text_model.predict_proba(x_test_hybrid_text)[:, 1]
        hybrid_text_missing_prob = hybrid_text_model.predict_proba(x_missing_hybrid_text)[:, 1]
        hybrid_text_blend_alpha = select_blend_weight(
            y_val,
            hybrid_blend_val_prob,
            hybrid_text_val_prob,
        )
        hybrid_text_blend_val_prob = blend_probabilities(
            hybrid_blend_val_prob,
            hybrid_text_val_prob,
            hybrid_text_blend_alpha,
        )
        hybrid_text_blend_test_prob = blend_probabilities(
            hybrid_blend_test_prob,
            hybrid_text_test_prob,
            hybrid_text_blend_alpha,
        )
        hybrid_text_blend_missing_prob = blend_probabilities(
            hybrid_blend_missing_prob,
            hybrid_text_missing_prob,
            hybrid_text_blend_alpha,
        )

        x_train_hybrid_text_retrieval = append_dense_features(x_train_hybrid_plus, train_text_stats)
        x_val_hybrid_text_retrieval = append_dense_features(x_val_hybrid_plus, val_text_stats)
        x_test_hybrid_text_retrieval = append_dense_features(x_test_hybrid_plus, test_text_stats)
        x_missing_hybrid_text_retrieval = append_dense_features(x_test_missing_hybrid_plus, missing_text_stats)

        hybrid_text_retrieval_model = train_classifier(x_train_hybrid_text_retrieval, y_train)
        hybrid_text_retrieval_val_prob = hybrid_text_retrieval_model.predict_proba(x_val_hybrid_text_retrieval)[:, 1]
        hybrid_text_retrieval_test_prob = hybrid_text_retrieval_model.predict_proba(x_test_hybrid_text_retrieval)[:, 1]
        hybrid_text_retrieval_missing_prob = hybrid_text_retrieval_model.predict_proba(x_missing_hybrid_text_retrieval)[:, 1]

        if smiles_embedder is not None:
            x_train_hybrid_smiles_text = append_dense_features(x_train_hybrid_pretrained, train_text_emb)
            x_val_hybrid_smiles_text = append_dense_features(x_val_hybrid_pretrained, val_text_emb)
            x_test_hybrid_smiles_text = append_dense_features(x_test_hybrid_pretrained, test_text_emb)
            x_missing_hybrid_smiles_text = append_dense_features(x_missing_hybrid_pretrained, missing_text_emb)

            hybrid_smiles_text_model = train_classifier(x_train_hybrid_smiles_text, y_train)
            hybrid_smiles_text_val_prob = hybrid_smiles_text_model.predict_proba(x_val_hybrid_smiles_text)[:, 1]
            hybrid_smiles_text_test_prob = hybrid_smiles_text_model.predict_proba(x_test_hybrid_smiles_text)[:, 1]
            hybrid_smiles_text_missing_prob = hybrid_smiles_text_model.predict_proba(x_missing_hybrid_smiles_text)[:, 1]
            hybrid_smiles_text_blend_alpha = select_blend_weight(
                y_val,
                hybrid_blend_val_prob,
                hybrid_smiles_text_val_prob,
            )
            hybrid_smiles_text_blend_val_prob = blend_probabilities(
                hybrid_blend_val_prob,
                hybrid_smiles_text_val_prob,
                hybrid_smiles_text_blend_alpha,
            )
            hybrid_smiles_text_blend_test_prob = blend_probabilities(
                hybrid_blend_test_prob,
                hybrid_smiles_text_test_prob,
                hybrid_smiles_text_blend_alpha,
            )
            hybrid_smiles_text_blend_missing_prob = blend_probabilities(
                hybrid_blend_missing_prob,
                hybrid_smiles_text_missing_prob,
                hybrid_smiles_text_blend_alpha,
            )

            x_train_hybrid_smiles_text_retrieval = append_dense_features(x_train_hybrid_pretrained, train_text_stats)
            x_val_hybrid_smiles_text_retrieval = append_dense_features(x_val_hybrid_pretrained, val_text_stats)
            x_test_hybrid_smiles_text_retrieval = append_dense_features(x_test_hybrid_pretrained, test_text_stats)
            x_missing_hybrid_smiles_text_retrieval = append_dense_features(x_missing_hybrid_pretrained, missing_text_stats)

            hybrid_smiles_text_retrieval_model = train_classifier(x_train_hybrid_smiles_text_retrieval, y_train)
            hybrid_smiles_text_retrieval_val_prob = hybrid_smiles_text_retrieval_model.predict_proba(x_val_hybrid_smiles_text_retrieval)[:, 1]
            hybrid_smiles_text_retrieval_test_prob = hybrid_smiles_text_retrieval_model.predict_proba(x_test_hybrid_smiles_text_retrieval)[:, 1]
            hybrid_smiles_text_retrieval_missing_prob = hybrid_smiles_text_retrieval_model.predict_proba(x_missing_hybrid_smiles_text_retrieval)[:, 1]

            strong_gate_val_x = build_gate_features(
                hybrid_blend_val_prob,
                hybrid_smiles_text_val_prob,
                np.hstack([val_morgan_stats, val_text_stats]).astype(np.float32),
                x_val_mask.masks,
            )
            strong_gate_test_x = build_gate_features(
                hybrid_blend_test_prob,
                hybrid_smiles_text_test_prob,
                np.hstack([test_morgan_stats, test_text_stats]).astype(np.float32),
                x_test_mask.masks,
            )
            strong_gate_missing_x = build_gate_features(
                hybrid_blend_missing_prob,
                hybrid_smiles_text_missing_prob,
                np.hstack([missing_morgan_stats, missing_text_stats]).astype(np.float32),
                x_test_missing_mask.masks,
            )
            strong_gate_model = fit_gate_model(strong_gate_val_x, y_val)
            strong_gate_val_prob = strong_gate_model.predict_proba(strong_gate_val_x)
            strong_gate_test_prob = strong_gate_model.predict_proba(strong_gate_test_x)
            strong_gate_missing_prob = strong_gate_model.predict_proba(strong_gate_missing_x)

            can_train_probe = (
                enable_interaction_probe
                and len(train_df) >= 8
                and len(val_df) >= 4
                and len(np.unique(y_train)) >= 2
                and len(np.unique(y_val)) >= 2
            )
            if can_train_probe:
                interaction_probe_config = interaction_probe_config or {}
                requested_text_field = str(interaction_probe_config.get("text_field", "text"))
                interaction_probe_text_field = (
                    requested_text_field
                    if requested_text_field in train_df.columns
                    else "text"
                )
                if interaction_probe_text_field == "text":
                    train_probe_text_emb = train_text_emb
                    val_probe_text_emb = val_text_emb
                    test_probe_text_emb = test_text_emb
                    missing_probe_text_emb = missing_text_emb
                    train_probe_text_stats = train_text_stats
                    val_probe_text_stats = val_text_stats
                    test_probe_text_stats = test_text_stats
                    missing_probe_text_stats = missing_text_stats
                else:
                    train_probe_text_emb = text_embedder.transform(train_df[interaction_probe_text_field])
                    val_probe_text_emb = text_embedder.transform(val_df[interaction_probe_text_field])
                    test_probe_text_emb = text_embedder.transform(test_df[interaction_probe_text_field])
                    missing_probe_text_emb = text_embedder.transform(stressed_test_df[interaction_probe_text_field])

                    train_probe_text_sparse = sparse.csr_matrix(train_probe_text_emb)
                    val_probe_text_sparse = sparse.csr_matrix(val_probe_text_emb)
                    test_probe_text_sparse = sparse.csr_matrix(test_probe_text_emb)
                    missing_probe_text_sparse = sparse.csr_matrix(missing_probe_text_emb)
                    probe_text_retrieval = RetrievalAugmentor(
                        n_neighbors=n_neighbors,
                        max_reference_size=retrieval_reference_size,
                    ).fit(train_probe_text_sparse, y_train)
                    train_probe_text_stats = probe_text_retrieval.transform_train(train_probe_text_sparse)
                    val_probe_text_stats = probe_text_retrieval.transform(val_probe_text_sparse)
                    test_probe_text_stats = probe_text_retrieval.transform(test_probe_text_sparse)
                    missing_probe_text_stats = probe_text_retrieval.transform(missing_probe_text_sparse)

                train_probe_aux = np.hstack([train_morgan_stats, train_mv_stats[:, smiles_slice], train_probe_text_stats]).astype(np.float32)
                val_probe_aux = np.hstack([val_morgan_stats, val_mv_stats[:, smiles_slice], val_probe_text_stats]).astype(np.float32)
                test_probe_aux = np.hstack([test_morgan_stats, test_mv_stats[:, smiles_slice], test_probe_text_stats]).astype(np.float32)
                missing_probe_aux = np.hstack([missing_morgan_stats, missing_mv_stats[:, smiles_slice], missing_probe_text_stats]).astype(np.float32)

                probe_bundle = train_interaction_probe(
                    train_smiles=train_smiles_emb,
                    train_text=train_probe_text_emb,
                    train_masks=x_train_mask.masks,
                    train_aux=train_probe_aux,
                    y_train=y_train,
                    val_smiles=val_smiles_emb,
                    val_text=val_probe_text_emb,
                    val_masks=x_val_mask.masks,
                    val_aux=val_probe_aux,
                    y_val=y_val,
                    device=interaction_probe_config.get("device"),
                    batch_size=int(interaction_probe_config.get("batch_size", 64)),
                    max_epochs=int(interaction_probe_config.get("max_epochs", 60)),
                    patience=int(interaction_probe_config.get("patience", 8)),
                    hidden_dim=int(interaction_probe_config.get("hidden_dim", 256)),
                    proj_dim=int(interaction_probe_config.get("proj_dim", 128)),
                    dropout=float(interaction_probe_config.get("dropout", 0.15)),
                    lr=float(interaction_probe_config.get("lr", 3e-4)),
                    weight_decay=float(interaction_probe_config.get("weight_decay", 3e-4)),
                    text_dropout_prob=float(interaction_probe_config.get("text_dropout_prob", 0.15)),
                    seed=int(interaction_probe_config.get("seed", 42)),
                )
                interaction_probe_best_epoch = probe_bundle.best_epoch
                interaction_probe_best_val_auprc = probe_bundle.best_val_auprc

                interaction_probe_val_prob = predict_interaction_probe(
                    bundle=probe_bundle,
                    smiles_emb=val_smiles_emb,
                    text_emb=val_probe_text_emb,
                    masks=x_val_mask.masks,
                    aux=val_probe_aux,
                    device=interaction_probe_config.get("device"),
                    batch_size=int(interaction_probe_config.get("batch_size", 64)),
                )
                interaction_probe_test_prob = predict_interaction_probe(
                    bundle=probe_bundle,
                    smiles_emb=test_smiles_emb,
                    text_emb=test_probe_text_emb,
                    masks=x_test_mask.masks,
                    aux=test_probe_aux,
                    device=interaction_probe_config.get("device"),
                    batch_size=int(interaction_probe_config.get("batch_size", 64)),
                )
                interaction_probe_missing_prob = predict_interaction_probe(
                    bundle=probe_bundle,
                    smiles_emb=missing_smiles_emb,
                    text_emb=missing_probe_text_emb,
                    masks=x_test_missing_mask.masks,
                    aux=missing_probe_aux,
                    device=interaction_probe_config.get("device"),
                    batch_size=int(interaction_probe_config.get("batch_size", 64)),
                )

                # Keep the probe branch definition fixed across datasets and splits.
                # Validation is used for ordinary early stopping and blend-weight tuning,
                # not to choose a different current-evidence architecture per split.
                interaction_probe_selected_baseline = "hybrid_plus_pretrained_smiles_text"
                selected_baseline_probs = {
                    "hybrid_blend_avg": (
                        hybrid_blend_val_prob,
                        hybrid_blend_test_prob,
                        hybrid_blend_missing_prob,
                    ),
                    "hybrid_plus_pretrained_smiles_text": (
                        hybrid_smiles_text_val_prob,
                        hybrid_smiles_text_test_prob,
                        hybrid_smiles_text_missing_prob,
                    ),
                }
                (
                    selected_baseline_val_prob,
                    selected_baseline_test_prob,
                    selected_baseline_missing_prob,
                ) = selected_baseline_probs[interaction_probe_selected_baseline]
                interaction_probe_blend_alpha = select_blend_weight(
                    y_val,
                    selected_baseline_val_prob,
                    interaction_probe_val_prob,
                )
                interaction_probe_blend_val_prob = blend_probabilities(
                    selected_baseline_val_prob,
                    interaction_probe_val_prob,
                    interaction_probe_blend_alpha,
                )
                interaction_probe_blend_test_prob = blend_probabilities(
                    selected_baseline_test_prob,
                    interaction_probe_test_prob,
                    interaction_probe_blend_alpha,
                )
                interaction_probe_blend_missing_prob = blend_probabilities(
                    selected_baseline_missing_prob,
                    interaction_probe_missing_prob,
                    interaction_probe_blend_alpha,
                )

    gate_val_x = build_gate_features(mask_val_prob, multiview_val_prob, val_mv_stats, x_val_mask.masks)
    gate_test_x = build_gate_features(mask_test_prob, multiview_test_prob, test_mv_stats, x_test_mask.masks)
    gate_missing_x = build_gate_features(
        mask_missing_prob,
        multiview_missing_prob,
        missing_mv_stats,
        x_test_missing_mask.masks,
    )
    gate_model = fit_gate_model(gate_val_x, y_val)
    gated_val_prob = gate_model.predict_proba(gate_val_x)
    gated_test_prob = gate_model.predict_proba(gate_test_x)
    gated_missing_prob = gate_model.predict_proba(gate_missing_x)
    if strong_gate_val_prob is not None:
        gated_strong_pretrained_blend_alpha = select_blend_weight(
            y_val,
            gated_val_prob,
            strong_gate_val_prob,
        )
        gated_strong_pretrained_val_prob = blend_probabilities(
            gated_val_prob,
            strong_gate_val_prob,
            gated_strong_pretrained_blend_alpha,
        )
        gated_strong_pretrained_test_prob = blend_probabilities(
            gated_test_prob,
            strong_gate_test_prob,
            gated_strong_pretrained_blend_alpha,
        )
        gated_strong_pretrained_missing_prob = blend_probabilities(
            gated_missing_prob,
            strong_gate_missing_prob,
            gated_strong_pretrained_blend_alpha,
        )
    if gated_strong_pretrained_val_prob is not None and hybrid_smiles_sequence_val_prob is not None:
        robust_sequence_anchor_alpha = ROBUST_SEQUENCE_ANCHOR_ALPHA
        robust_sequence_anchor_val_prob = blend_probabilities(
            gated_strong_pretrained_val_prob,
            hybrid_smiles_sequence_val_prob,
            robust_sequence_anchor_alpha,
        )
        robust_sequence_anchor_test_prob = blend_probabilities(
            gated_strong_pretrained_test_prob,
            hybrid_smiles_sequence_test_prob,
            robust_sequence_anchor_alpha,
        )
        robust_sequence_anchor_missing_prob = blend_probabilities(
            gated_strong_pretrained_missing_prob,
            hybrid_smiles_sequence_missing_prob,
            robust_sequence_anchor_alpha,
        )
        sequence_gate_val_x = build_gate_features(
            gated_strong_pretrained_val_prob,
            hybrid_smiles_sequence_val_prob,
            np.hstack([val_morgan_stats, val_text_stats]).astype(np.float32),
            x_val_mask.masks,
        )
        sequence_gate_test_x = build_gate_features(
            gated_strong_pretrained_test_prob,
            hybrid_smiles_sequence_test_prob,
            np.hstack([test_morgan_stats, test_text_stats]).astype(np.float32),
            x_test_mask.masks,
        )
        sequence_gate_missing_x = build_gate_features(
            gated_strong_pretrained_missing_prob,
            hybrid_smiles_sequence_missing_prob,
            np.hstack([missing_morgan_stats, missing_text_stats]).astype(np.float32),
            x_test_missing_mask.masks,
        )
        sequence_gate_model = fit_gate_model(sequence_gate_val_x, y_val)
        sequence_gate_val_prob = sequence_gate_model.predict_proba(sequence_gate_val_x)
        sequence_gate_test_prob = sequence_gate_model.predict_proba(sequence_gate_test_x)
        sequence_gate_missing_prob = sequence_gate_model.predict_proba(sequence_gate_missing_x)
    if gated_val_prob is not None and hybrid_smiles_sequence_val_prob is not None:
        retrieval_sequence_bridge_alpha = RETRIEVAL_SEQUENCE_BRIDGE_ALPHA
        retrieval_sequence_bridge_val_prob = blend_probabilities(
            gated_val_prob,
            hybrid_smiles_sequence_val_prob,
            retrieval_sequence_bridge_alpha,
        )
        retrieval_sequence_bridge_test_prob = blend_probabilities(
            gated_test_prob,
            hybrid_smiles_sequence_test_prob,
            retrieval_sequence_bridge_alpha,
        )
        retrieval_sequence_bridge_missing_prob = blend_probabilities(
            gated_missing_prob,
            hybrid_smiles_sequence_missing_prob,
            retrieval_sequence_bridge_alpha,
        )
    if gated_strong_pretrained_val_prob is not None and interaction_probe_blend_val_prob is not None:
        robust_probe_anchor_alpha = ROBUST_PROBE_ANCHOR_ALPHA
        robust_probe_anchor_val_prob = blend_probabilities(
            gated_strong_pretrained_val_prob,
            interaction_probe_blend_val_prob,
            robust_probe_anchor_alpha,
        )
        robust_probe_anchor_test_prob = blend_probabilities(
            gated_strong_pretrained_test_prob,
            interaction_probe_blend_test_prob,
            robust_probe_anchor_alpha,
        )
        robust_probe_anchor_missing_prob = blend_probabilities(
            gated_strong_pretrained_missing_prob,
            interaction_probe_blend_missing_prob,
            robust_probe_anchor_alpha,
        )
        interaction_gate_blend_alpha = select_blend_weight(
            y_val,
            gated_strong_pretrained_val_prob,
            interaction_probe_blend_val_prob,
        )
        interaction_gate_val_prob = blend_probabilities(
            gated_strong_pretrained_val_prob,
            interaction_probe_blend_val_prob,
            interaction_gate_blend_alpha,
        )
        interaction_gate_test_prob = blend_probabilities(
            gated_strong_pretrained_test_prob,
            interaction_probe_blend_test_prob,
            interaction_gate_blend_alpha,
        )
        interaction_gate_missing_prob = blend_probabilities(
            gated_strong_pretrained_missing_prob,
            interaction_probe_blend_missing_prob,
            interaction_gate_blend_alpha,
        )

    if mask_val_prob is not None:
        arbitration = _fit_conflict_aware_arbitration(
            y_train=y_train,
            y_val=y_val,
            current_val=mask_val_prob,
            retrieval_val=retrieval_evidence_val_prob,
            current_test=mask_test_prob,
            retrieval_test=retrieval_evidence_test_prob,
            current_missing=mask_missing_prob,
            retrieval_missing=retrieval_evidence_missing_prob,
            val_stats=val_mv_stats,
            test_stats=test_mv_stats,
            missing_stats=missing_mv_stats,
            val_masks=x_val_mask.masks,
            test_masks=x_test_mask.masks,
            missing_masks=x_test_missing_mask.masks,
        )
        mirage_full_val_prob = arbitration["val"]["full"]
        mirage_full_test_prob = arbitration["test"]["full"]
        mirage_full_missing_prob = arbitration["missing"]["full"]
        mirage_no_gate_val_prob = arbitration["val"]["no_gate"]
        mirage_no_gate_test_prob = arbitration["test"]["no_gate"]
        mirage_no_gate_missing_prob = arbitration["missing"]["no_gate"]
        mirage_no_probe_val_prob = arbitration["val"]["no_probe"]
        mirage_no_probe_test_prob = arbitration["test"]["no_probe"]
        mirage_no_probe_missing_prob = arbitration["missing"]["no_probe"]
        mirage_no_anchor_val_prob = arbitration["val"]["no_anchor"]
        mirage_no_anchor_test_prob = arbitration["test"]["no_anchor"]
        mirage_no_anchor_missing_prob = arbitration["missing"]["no_anchor"]
        mirage_gate_val = arbitration["val"]["gamma"]
        mirage_gate_test = arbitration["test"]["gamma"]
        mirage_gate_missing = arbitration["missing"]["gamma"]
        mirage_reliability_val = arbitration["val"]["reliability"]
        mirage_reliability_test = arbitration["test"]["reliability"]
        mirage_reliability_missing = arbitration["missing"]["reliability"]
        mirage_anchor_probability = float(arbitration["anchor"])
        mirage_anchor_strength = float(arbitration["anchor_strength"])
        mirage_probe_oof_accuracy = float(arbitration["probe_oof_accuracy"])

    fullsuite_candidates = {
        "hybrid_blend_avg": (
            hybrid_blend_val_prob,
            hybrid_blend_test_prob,
            hybrid_blend_missing_prob,
        ),
        "gated_retrieval": (
            gated_val_prob,
            gated_test_prob,
            gated_missing_prob,
        ),
    }
    if hybrid_smiles_text_val_prob is not None:
        fullsuite_candidates["hybrid_plus_pretrained_smiles_text"] = (
            hybrid_smiles_text_val_prob,
            hybrid_smiles_text_test_prob,
            hybrid_smiles_text_missing_prob,
        )
    if hybrid_smiles_sequence_val_prob is not None:
        fullsuite_candidates["hybrid_plus_pretrained_smiles_sequence"] = (
            hybrid_smiles_sequence_val_prob,
            hybrid_smiles_sequence_test_prob,
            hybrid_smiles_sequence_missing_prob,
        )
    if strong_gate_val_prob is not None:
        fullsuite_candidates["strong_pretrained_gate"] = (
            strong_gate_val_prob,
            strong_gate_test_prob,
            strong_gate_missing_prob,
        )
    if gated_strong_pretrained_val_prob is not None:
        fullsuite_candidates["gated_strong_pretrained_tuned"] = (
            gated_strong_pretrained_val_prob,
            gated_strong_pretrained_test_prob,
            gated_strong_pretrained_missing_prob,
        )
    if robust_sequence_anchor_val_prob is not None:
        fullsuite_candidates["robust_sequence_anchor"] = (
            robust_sequence_anchor_val_prob,
            robust_sequence_anchor_test_prob,
            robust_sequence_anchor_missing_prob,
        )
    if robust_probe_anchor_val_prob is not None:
        fullsuite_candidates["robust_probe_anchor"] = (
            robust_probe_anchor_val_prob,
            robust_probe_anchor_test_prob,
            robust_probe_anchor_missing_prob,
        )
    if retrieval_sequence_bridge_val_prob is not None:
        fullsuite_candidates["retrieval_sequence_bridge"] = (
            retrieval_sequence_bridge_val_prob,
            retrieval_sequence_bridge_test_prob,
            retrieval_sequence_bridge_missing_prob,
        )
    if sequence_gate_val_prob is not None:
        fullsuite_candidates["sequence_gate_tuned"] = (
            sequence_gate_val_prob,
            sequence_gate_test_prob,
            sequence_gate_missing_prob,
        )
    if interaction_probe_blend_val_prob is not None:
        fullsuite_candidates["interaction_probe_smiles_text_blend"] = (
            interaction_probe_blend_val_prob,
            interaction_probe_blend_test_prob,
            interaction_probe_blend_missing_prob,
        )
    if interaction_gate_val_prob is not None:
        fullsuite_candidates["interaction_gate_tuned"] = (
            interaction_gate_val_prob,
            interaction_gate_test_prob,
            interaction_gate_missing_prob,
        )
    fullsuite_val_select_name = select_best_candidate(
        y_val,
        {name: values[0] for name, values in fullsuite_candidates.items()},
    )
    (
        fullsuite_val_select_val_prob,
        fullsuite_val_select_test_prob,
        fullsuite_val_select_missing_prob,
    ) = fullsuite_candidates[fullsuite_val_select_name]

    model_prob_triplets = {
        "mirage_full": (mirage_full_val_prob, mirage_full_test_prob, mirage_full_missing_prob),
        "mirage_w_o_gate": (mirage_no_gate_val_prob, mirage_no_gate_test_prob, mirage_no_gate_missing_prob),
        "mirage_w_o_probe": (mirage_no_probe_val_prob, mirage_no_probe_test_prob, mirage_no_probe_missing_prob),
        "mirage_w_o_anchor": (mirage_no_anchor_val_prob, mirage_no_anchor_test_prob, mirage_no_anchor_missing_prob),
        "no_mask": (no_mask_val_prob, no_mask_test_prob, no_mask_missing_prob),
        "mask": (mask_val_prob, mask_test_prob, mask_missing_prob),
        "retrieval": (retrieval_val_prob, retrieval_test_prob, retrieval_missing_prob),
        "smiles_only_retrieval": (smiles_retrieval_val_prob, smiles_retrieval_test_prob, smiles_retrieval_missing_prob),
        "morgan_smiles_retrieval": (morgan_smiles_val_prob, morgan_smiles_test_prob, morgan_smiles_missing_prob),
        "hybrid_smiles_retrieval": (hybrid_smiles_val_prob, hybrid_smiles_test_prob, hybrid_smiles_missing_prob),
        "hybrid_plus_morgan_bits": (hybrid_plus_val_prob, hybrid_plus_test_prob, hybrid_plus_missing_prob),
        "hybrid_blend_avg": (hybrid_blend_val_prob, hybrid_blend_test_prob, hybrid_blend_missing_prob),
        "multiview_retrieval": (multiview_val_prob, multiview_test_prob, multiview_missing_prob),
        "historical_retrieval_evidence": (
            retrieval_evidence_val_prob,
            retrieval_evidence_test_prob,
            retrieval_evidence_missing_prob,
        ),
        "gated_retrieval": (gated_val_prob, gated_test_prob, gated_missing_prob),
        "pretrained_smiles_dense": (pretrained_dense_val_prob, pretrained_dense_test_prob, pretrained_dense_missing_prob),
        "pretrained_smiles_retrieval": (
            pretrained_retrieval_val_prob,
            pretrained_retrieval_test_prob,
            pretrained_retrieval_missing_prob,
        ),
        "hybrid_plus_pretrained_smiles": (
            hybrid_pretrained_val_prob,
            hybrid_pretrained_test_prob,
            hybrid_pretrained_missing_prob,
        ),
        "hybrid_blend_pretrained_tuned": (
            hybrid_pretrained_blend_val_prob,
            hybrid_pretrained_blend_test_prob,
            hybrid_pretrained_blend_missing_prob,
        ),
        "hybrid_pretrained_val_select": (
            hybrid_pretrained_select_val_prob,
            hybrid_pretrained_select_test_prob,
            hybrid_pretrained_select_missing_prob,
        ),
        "hybrid_plus_pretrained_text": (hybrid_text_val_prob, hybrid_text_test_prob, hybrid_text_missing_prob),
        "hybrid_blend_pretrained_text_tuned": (
            hybrid_text_blend_val_prob,
            hybrid_text_blend_test_prob,
            hybrid_text_blend_missing_prob,
        ),
        "hybrid_plus_pretrained_smiles_text": (
            hybrid_smiles_text_val_prob,
            hybrid_smiles_text_test_prob,
            hybrid_smiles_text_missing_prob,
        ),
        "hybrid_blend_pretrained_smiles_text_tuned": (
            hybrid_smiles_text_blend_val_prob,
            hybrid_smiles_text_blend_test_prob,
            hybrid_smiles_text_blend_missing_prob,
        ),
        "strong_pretrained_gate": (strong_gate_val_prob, strong_gate_test_prob, strong_gate_missing_prob),
        "gated_strong_pretrained_tuned": (
            gated_strong_pretrained_val_prob,
            gated_strong_pretrained_test_prob,
            gated_strong_pretrained_missing_prob,
        ),
        "robust_sequence_anchor": (
            robust_sequence_anchor_val_prob,
            robust_sequence_anchor_test_prob,
            robust_sequence_anchor_missing_prob,
        ),
        "robust_probe_anchor": (
            robust_probe_anchor_val_prob,
            robust_probe_anchor_test_prob,
            robust_probe_anchor_missing_prob,
        ),
        "retrieval_sequence_bridge": (
            retrieval_sequence_bridge_val_prob,
            retrieval_sequence_bridge_test_prob,
            retrieval_sequence_bridge_missing_prob,
        ),
        "sequence_gate_tuned": (sequence_gate_val_prob, sequence_gate_test_prob, sequence_gate_missing_prob),
        "interaction_gate_tuned": (interaction_gate_val_prob, interaction_gate_test_prob, interaction_gate_missing_prob),
        "fullsuite_val_select": (
            fullsuite_val_select_val_prob,
            fullsuite_val_select_test_prob,
            fullsuite_val_select_missing_prob,
        ),
        "interaction_probe_smiles_text": (
            interaction_probe_val_prob,
            interaction_probe_test_prob,
            interaction_probe_missing_prob,
        ),
        "interaction_probe_smiles_text_blend": (
            interaction_probe_blend_val_prob,
            interaction_probe_blend_test_prob,
            interaction_probe_blend_missing_prob,
        ),
        "pretrained_text_retrieval": (
            pretrained_text_retrieval_val_prob,
            pretrained_text_retrieval_test_prob,
            pretrained_text_retrieval_missing_prob,
        ),
        "hybrid_plus_pretrained_text_retrieval": (
            hybrid_text_retrieval_val_prob,
            hybrid_text_retrieval_test_prob,
            hybrid_text_retrieval_missing_prob,
        ),
        "hybrid_plus_pretrained_smiles_text_retrieval": (
            hybrid_smiles_text_retrieval_val_prob,
            hybrid_smiles_text_retrieval_test_prob,
            hybrid_smiles_text_retrieval_missing_prob,
        ),
        "hybrid_plus_pretrained_sequence": (
            hybrid_sequence_val_prob,
            hybrid_sequence_test_prob,
            hybrid_sequence_missing_prob,
        ),
        "hybrid_plus_pretrained_smiles_sequence": (
            hybrid_smiles_sequence_val_prob,
            hybrid_smiles_sequence_test_prob,
            hybrid_smiles_sequence_missing_prob,
        ),
    }

    validation_predictions = pd.DataFrame(
        {
            "sample_id": val_df["sample_id"] if "sample_id" in val_df.columns else [f"row_{i}" for i in range(len(val_df))],
            "drug_id": val_df["drug_id"] if "drug_id" in val_df.columns else "",
            "target_id": val_df["target_id"] if "target_id" in val_df.columns else "",
            "assay_id": val_df["assay_id"] if "assay_id" in val_df.columns else "",
            "document_year": val_df["document_year"] if "document_year" in val_df.columns else pd.Series([pd.NA] * len(val_df)),
            "label": y_val,
            "label_raw": val_df["label_raw"] if "label_raw" in val_df.columns else pd.Series([np.nan] * len(val_df)),
            "smiles": val_df["smiles"] if "smiles" in val_df.columns else "",
            "sequence": val_df["sequence"] if "sequence" in val_df.columns else "",
            "text": val_df["text"] if "text" in val_df.columns else "",
            "mirage_gate_weight": mirage_gate_val,
            "mirage_reliability": mirage_reliability_val,
            **{f"{name}_prob": payload[0] for name, payload in model_prob_triplets.items()},
        }
    )

    predictions = pd.DataFrame(
        {
            "sample_id": test_df["sample_id"] if "sample_id" in test_df.columns else [f"row_{i}" for i in range(len(test_df))],
            "drug_id": test_df["drug_id"] if "drug_id" in test_df.columns else "",
            "target_id": test_df["target_id"] if "target_id" in test_df.columns else "",
            "assay_id": test_df["assay_id"] if "assay_id" in test_df.columns else "",
            "document_year": test_df["document_year"] if "document_year" in test_df.columns else pd.Series([pd.NA] * len(test_df)),
            "label": y_test,
            "label_raw": test_df["label_raw"] if "label_raw" in test_df.columns else pd.Series([np.nan] * len(test_df)),
            "smiles": test_df["smiles"] if "smiles" in test_df.columns else "",
            "sequence": test_df["sequence"] if "sequence" in test_df.columns else "",
            "text": test_df["text"] if "text" in test_df.columns else "",
            "mirage_gate_weight_clean": mirage_gate_test,
            "mirage_gate_weight_missing": mirage_gate_missing,
            "mirage_reliability_clean": mirage_reliability_test,
            "mirage_reliability_missing": mirage_reliability_missing,
            **{f"{name}_prob_clean": payload[1] for name, payload in model_prob_triplets.items()},
            **{f"{name}_prob_missing": payload[2] for name, payload in model_prob_triplets.items()},
            "text_missing": stressed_test_df["text"].eq(""),
            "sequence_missing": stressed_test_df["sequence"].eq(""),
        }
    )

    models = {
        "no_mask": {
            "val": evaluate_binary(y_val, no_mask_val_prob),
            "test_clean": evaluate_binary(y_test, no_mask_test_prob),
            "test_missing": evaluate_binary(y_test, no_mask_missing_prob),
        },
        "mask": {
            "val": evaluate_binary(y_val, mask_val_prob),
            "test_clean": evaluate_binary(y_test, mask_test_prob),
            "test_missing": evaluate_binary(y_test, mask_missing_prob),
        },
        "retrieval": {
            "val": evaluate_binary(y_val, retrieval_val_prob),
            "test_clean": evaluate_binary(y_test, retrieval_test_prob),
            "test_missing": evaluate_binary(y_test, retrieval_missing_prob),
        },
        "smiles_only_retrieval": {
            "val": evaluate_binary(y_val, smiles_retrieval_val_prob),
            "test_clean": evaluate_binary(y_test, smiles_retrieval_test_prob),
            "test_missing": evaluate_binary(y_test, smiles_retrieval_missing_prob),
        },
        "morgan_smiles_retrieval": {
            "val": evaluate_binary(y_val, morgan_smiles_val_prob),
            "test_clean": evaluate_binary(y_test, morgan_smiles_test_prob),
            "test_missing": evaluate_binary(y_test, morgan_smiles_missing_prob),
        },
        "hybrid_smiles_retrieval": {
            "val": evaluate_binary(y_val, hybrid_smiles_val_prob),
            "test_clean": evaluate_binary(y_test, hybrid_smiles_test_prob),
            "test_missing": evaluate_binary(y_test, hybrid_smiles_missing_prob),
        },
        "hybrid_plus_morgan_bits": {
            "val": evaluate_binary(y_val, hybrid_plus_val_prob),
            "test_clean": evaluate_binary(y_test, hybrid_plus_test_prob),
            "test_missing": evaluate_binary(y_test, hybrid_plus_missing_prob),
        },
        "hybrid_blend_avg": {
            "val": evaluate_binary(y_val, hybrid_blend_val_prob),
            "test_clean": evaluate_binary(y_test, hybrid_blend_test_prob),
            "test_missing": evaluate_binary(y_test, hybrid_blend_missing_prob),
        },
        "multiview_retrieval": {
            "val": evaluate_binary(y_val, multiview_val_prob),
            "test_clean": evaluate_binary(y_test, multiview_test_prob),
            "test_missing": evaluate_binary(y_test, multiview_missing_prob),
        },
        "historical_retrieval_evidence": {
            "val": evaluate_binary(y_val, retrieval_evidence_val_prob),
            "test_clean": evaluate_binary(y_test, retrieval_evidence_test_prob),
            "test_missing": evaluate_binary(y_test, retrieval_evidence_missing_prob),
        },
        "gated_retrieval": {
            "val": evaluate_binary(y_val, gated_val_prob),
            "test_clean": evaluate_binary(y_test, gated_test_prob),
            "test_missing": evaluate_binary(y_test, gated_missing_prob),
        },
        "fullsuite_val_select": {
            "val": evaluate_binary(y_val, fullsuite_val_select_val_prob),
            "test_clean": evaluate_binary(y_test, fullsuite_val_select_test_prob),
            "test_missing": evaluate_binary(y_test, fullsuite_val_select_missing_prob),
            "selected_model": fullsuite_val_select_name,
        },
    }
    if mirage_full_val_prob is not None:
        models.update(
            {
                "mirage_full": {
                    "val": evaluate_binary(y_val, mirage_full_val_prob),
                    "test_clean": evaluate_binary(y_test, mirage_full_test_prob),
                    "test_missing": evaluate_binary(y_test, mirage_full_missing_prob),
                    "anchor_probability": mirage_anchor_probability,
                    "anchor_strength": mirage_anchor_strength,
                    "probe_oof_accuracy": mirage_probe_oof_accuracy,
                    "definition": "mask-aware direct current branch, training-memory branch, cross-fitted arbitration gate, reliability probe, and validation-calibrated prevalence anchor",
                },
                "mirage_w_o_gate": {
                    "val": evaluate_binary(y_val, mirage_no_gate_val_prob),
                    "test_clean": evaluate_binary(y_test, mirage_no_gate_test_prob),
                    "test_missing": evaluate_binary(y_test, mirage_no_gate_missing_prob),
                },
                "mirage_w_o_probe": {
                    "val": evaluate_binary(y_val, mirage_no_probe_val_prob),
                    "test_clean": evaluate_binary(y_test, mirage_no_probe_test_prob),
                    "test_missing": evaluate_binary(y_test, mirage_no_probe_missing_prob),
                },
                "mirage_w_o_anchor": {
                    "val": evaluate_binary(y_val, mirage_no_anchor_val_prob),
                    "test_clean": evaluate_binary(y_test, mirage_no_anchor_test_prob),
                    "test_missing": evaluate_binary(y_test, mirage_no_anchor_missing_prob),
                },
            }
        )
    if smiles_embedder is not None:
        models.update(
            {
                "pretrained_smiles_dense": {
                    "val": evaluate_binary(y_val, pretrained_dense_val_prob),
                    "test_clean": evaluate_binary(y_test, pretrained_dense_test_prob),
                    "test_missing": evaluate_binary(y_test, pretrained_dense_missing_prob),
                },
                "pretrained_smiles_retrieval": {
                    "val": evaluate_binary(y_val, pretrained_retrieval_val_prob),
                    "test_clean": evaluate_binary(y_test, pretrained_retrieval_test_prob),
                    "test_missing": evaluate_binary(y_test, pretrained_retrieval_missing_prob),
                },
                "hybrid_plus_pretrained_smiles": {
                    "val": evaluate_binary(y_val, hybrid_pretrained_val_prob),
                    "test_clean": evaluate_binary(y_test, hybrid_pretrained_test_prob),
                    "test_missing": evaluate_binary(y_test, hybrid_pretrained_missing_prob),
                },
                "hybrid_blend_pretrained_tuned": {
                    "val": evaluate_binary(y_val, hybrid_pretrained_blend_val_prob),
                    "test_clean": evaluate_binary(y_test, hybrid_pretrained_blend_test_prob),
                    "test_missing": evaluate_binary(y_test, hybrid_pretrained_blend_missing_prob),
                    "blend_alpha": hybrid_pretrained_blend_alpha,
                },
                "hybrid_pretrained_val_select": {
                    "val": evaluate_binary(y_val, hybrid_pretrained_select_val_prob),
                    "test_clean": evaluate_binary(y_test, hybrid_pretrained_select_test_prob),
                    "test_missing": evaluate_binary(y_test, hybrid_pretrained_select_missing_prob),
                    "selected_model": hybrid_pretrained_selector_name,
                },
            }
        )
    if sequence_embedder is not None:
        models.update(
            {
                "hybrid_plus_pretrained_sequence": {
                    "val": evaluate_binary(y_val, hybrid_sequence_val_prob),
                    "test_clean": evaluate_binary(y_test, hybrid_sequence_test_prob),
                    "test_missing": evaluate_binary(y_test, hybrid_sequence_missing_prob),
                },
            }
        )
        if smiles_embedder is not None:
            models.update(
                {
                    "hybrid_plus_pretrained_smiles_sequence": {
                        "val": evaluate_binary(y_val, hybrid_smiles_sequence_val_prob),
                        "test_clean": evaluate_binary(y_test, hybrid_smiles_sequence_test_prob),
                        "test_missing": evaluate_binary(y_test, hybrid_smiles_sequence_missing_prob),
                    },
                }
            )
    if text_embedder is not None:
        models.update(
            {
                "hybrid_plus_pretrained_text": {
                    "val": evaluate_binary(y_val, hybrid_text_val_prob),
                    "test_clean": evaluate_binary(y_test, hybrid_text_test_prob),
                    "test_missing": evaluate_binary(y_test, hybrid_text_missing_prob),
                },
                "hybrid_blend_pretrained_text_tuned": {
                    "val": evaluate_binary(y_val, hybrid_text_blend_val_prob),
                    "test_clean": evaluate_binary(y_test, hybrid_text_blend_test_prob),
                    "test_missing": evaluate_binary(y_test, hybrid_text_blend_missing_prob),
                    "blend_alpha": hybrid_text_blend_alpha,
                },
                "pretrained_text_retrieval": {
                    "val": evaluate_binary(y_val, pretrained_text_retrieval_val_prob),
                    "test_clean": evaluate_binary(y_test, pretrained_text_retrieval_test_prob),
                    "test_missing": evaluate_binary(y_test, pretrained_text_retrieval_missing_prob),
                },
                "hybrid_plus_pretrained_text_retrieval": {
                    "val": evaluate_binary(y_val, hybrid_text_retrieval_val_prob),
                    "test_clean": evaluate_binary(y_test, hybrid_text_retrieval_test_prob),
                    "test_missing": evaluate_binary(y_test, hybrid_text_retrieval_missing_prob),
                },
            }
        )
        if smiles_embedder is not None:
            models.update(
                {
                    "hybrid_plus_pretrained_smiles_text": {
                        "val": evaluate_binary(y_val, hybrid_smiles_text_val_prob),
                        "test_clean": evaluate_binary(y_test, hybrid_smiles_text_test_prob),
                        "test_missing": evaluate_binary(y_test, hybrid_smiles_text_missing_prob),
                    },
                    "hybrid_blend_pretrained_smiles_text_tuned": {
                        "val": evaluate_binary(y_val, hybrid_smiles_text_blend_val_prob),
                        "test_clean": evaluate_binary(y_test, hybrid_smiles_text_blend_test_prob),
                        "test_missing": evaluate_binary(y_test, hybrid_smiles_text_blend_missing_prob),
                        "blend_alpha": hybrid_smiles_text_blend_alpha,
                    },
                    "hybrid_plus_pretrained_smiles_text_retrieval": {
                        "val": evaluate_binary(y_val, hybrid_smiles_text_retrieval_val_prob),
                        "test_clean": evaluate_binary(y_test, hybrid_smiles_text_retrieval_test_prob),
                        "test_missing": evaluate_binary(y_test, hybrid_smiles_text_retrieval_missing_prob),
                    },
                    "strong_pretrained_gate": {
                        "val": evaluate_binary(y_val, strong_gate_val_prob),
                        "test_clean": evaluate_binary(y_test, strong_gate_test_prob),
                        "test_missing": evaluate_binary(y_test, strong_gate_missing_prob),
                    },
                    "gated_strong_pretrained_tuned": {
                        "val": evaluate_binary(y_val, gated_strong_pretrained_val_prob),
                        "test_clean": evaluate_binary(y_test, gated_strong_pretrained_test_prob),
                        "test_missing": evaluate_binary(y_test, gated_strong_pretrained_missing_prob),
                        "blend_alpha": gated_strong_pretrained_blend_alpha,
                    },
                }
            )
            if robust_sequence_anchor_val_prob is not None:
                models["robust_sequence_anchor"] = {
                    "val": evaluate_binary(y_val, robust_sequence_anchor_val_prob),
                    "test_clean": evaluate_binary(y_test, robust_sequence_anchor_test_prob),
                    "test_missing": evaluate_binary(y_test, robust_sequence_anchor_missing_prob),
                    "blend_alpha": robust_sequence_anchor_alpha,
                }
            if robust_probe_anchor_val_prob is not None:
                models["robust_probe_anchor"] = {
                    "val": evaluate_binary(y_val, robust_probe_anchor_val_prob),
                    "test_clean": evaluate_binary(y_test, robust_probe_anchor_test_prob),
                    "test_missing": evaluate_binary(y_test, robust_probe_anchor_missing_prob),
                    "blend_alpha": robust_probe_anchor_alpha,
                }
            if retrieval_sequence_bridge_val_prob is not None:
                models["retrieval_sequence_bridge"] = {
                    "val": evaluate_binary(y_val, retrieval_sequence_bridge_val_prob),
                    "test_clean": evaluate_binary(y_test, retrieval_sequence_bridge_test_prob),
                    "test_missing": evaluate_binary(y_test, retrieval_sequence_bridge_missing_prob),
                    "blend_alpha": retrieval_sequence_bridge_alpha,
                }
            if sequence_gate_val_prob is not None:
                models["sequence_gate_tuned"] = {
                    "val": evaluate_binary(y_val, sequence_gate_val_prob),
                    "test_clean": evaluate_binary(y_test, sequence_gate_test_prob),
                    "test_missing": evaluate_binary(y_test, sequence_gate_missing_prob),
                }
            if interaction_gate_val_prob is not None:
                models["interaction_gate_tuned"] = {
                    "val": evaluate_binary(y_val, interaction_gate_val_prob),
                    "test_clean": evaluate_binary(y_test, interaction_gate_test_prob),
                    "test_missing": evaluate_binary(y_test, interaction_gate_missing_prob),
                    "blend_alpha": interaction_gate_blend_alpha,
                }
            if interaction_probe_val_prob is not None:
                models.update(
                    {
                        "interaction_probe_smiles_text": {
                            "val": evaluate_binary(y_val, interaction_probe_val_prob),
                            "test_clean": evaluate_binary(y_test, interaction_probe_test_prob),
                            "test_missing": evaluate_binary(y_test, interaction_probe_missing_prob),
                            "best_epoch": interaction_probe_best_epoch,
                            "best_val_auprc": interaction_probe_best_val_auprc,
                            "text_field": interaction_probe_text_field,
                        },
                        "interaction_probe_smiles_text_blend": {
                            "val": evaluate_binary(y_val, interaction_probe_blend_val_prob),
                            "test_clean": evaluate_binary(y_test, interaction_probe_blend_test_prob),
                            "test_missing": evaluate_binary(y_test, interaction_probe_blend_missing_prob),
                            "blend_alpha": interaction_probe_blend_alpha,
                            "selected_baseline": interaction_probe_selected_baseline,
                            "text_field": interaction_probe_text_field,
                        },
                    }
                )

    return {
        "models": models,
        "predictions": predictions,
        "validation_predictions": validation_predictions,
    }


def run_benchmark_shortlist_suite(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stressed_test_df: pd.DataFrame,
    n_neighbors: int = 8,
    smiles_embedder=None,
    text_embedder=None,
    retrieval_reference_size: int | None = None,
) -> dict:
    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()

    mask_feat = MultiModalFeaturizer(include_masks=True)
    morgan_feat = MorganFingerprintFeaturizer(n_bits=2048, radius=2)

    x_train_mask = mask_feat.transform(train_df)
    x_val_mask = mask_feat.transform(val_df)
    x_test_mask = mask_feat.transform(test_df)
    x_test_missing_mask = mask_feat.transform(stressed_test_df)

    mask_model = train_classifier(x_train_mask.matrix, y_train)
    mask_val_prob = mask_model.predict_proba(x_val_mask.matrix)[:, 1]
    mask_test_prob = mask_model.predict_proba(x_test_mask.matrix)[:, 1]
    mask_missing_prob = mask_model.predict_proba(x_test_missing_mask.matrix)[:, 1]

    retrieval = RetrievalAugmentor(
        n_neighbors=n_neighbors,
        max_reference_size=retrieval_reference_size,
    ).fit(x_train_mask.matrix, y_train)
    x_train_aug = append_dense_features(x_train_mask.matrix, retrieval.transform_train(x_train_mask.matrix))
    x_val_aug = append_dense_features(x_val_mask.matrix, retrieval.transform(x_val_mask.matrix))
    x_test_aug = append_dense_features(x_test_mask.matrix, retrieval.transform(x_test_mask.matrix))
    x_test_missing_aug = append_dense_features(
        x_test_missing_mask.matrix,
        retrieval.transform(x_test_missing_mask.matrix),
    )
    retrieval_model = train_classifier(x_train_aug, y_train)
    retrieval_val_prob = retrieval_model.predict_proba(x_val_aug)[:, 1]
    retrieval_test_prob = retrieval_model.predict_proba(x_test_aug)[:, 1]
    retrieval_missing_prob = retrieval_model.predict_proba(x_test_missing_aug)[:, 1]

    multiview_retrieval = MultiViewRetrievalAugmentor(
        n_neighbors=n_neighbors,
        max_reference_size=retrieval_reference_size,
    ).fit(x_train_mask, y_train)
    train_mv_stats = multiview_retrieval.transform_train(x_train_mask)
    val_mv_stats = multiview_retrieval.transform(x_val_mask)
    test_mv_stats = multiview_retrieval.transform(x_test_mask)
    missing_mv_stats = multiview_retrieval.transform(x_test_missing_mask)
    smiles_idx = multiview_retrieval.view_names.index("smiles")
    smiles_slice = slice(
        smiles_idx * multiview_retrieval.stats_dim,
        (smiles_idx + 1) * multiview_retrieval.stats_dim,
    )

    x_train_smiles = append_dense_features(x_train_mask.matrix, train_mv_stats[:, smiles_slice])
    x_val_smiles = append_dense_features(x_val_mask.matrix, val_mv_stats[:, smiles_slice])
    x_test_smiles = append_dense_features(x_test_mask.matrix, test_mv_stats[:, smiles_slice])
    x_test_missing_smiles = append_dense_features(
        x_test_missing_mask.matrix,
        missing_mv_stats[:, smiles_slice],
    )

    x_train_morgan = morgan_feat.transform(train_df["smiles"])
    x_val_morgan = morgan_feat.transform(val_df["smiles"])
    x_test_morgan = morgan_feat.transform(test_df["smiles"])
    x_test_missing_morgan = morgan_feat.transform(stressed_test_df["smiles"])

    morgan_retrieval = TanimotoRetrievalAugmentor(
        n_neighbors=n_neighbors,
        max_reference_size=retrieval_reference_size,
    ).fit(x_train_morgan, y_train)
    train_morgan_stats = morgan_retrieval.transform_train(x_train_morgan)
    val_morgan_stats = morgan_retrieval.transform(x_val_morgan)
    test_morgan_stats = morgan_retrieval.transform(x_test_morgan)
    missing_morgan_stats = morgan_retrieval.transform(x_test_missing_morgan)

    x_train_hybrid = append_dense_features(x_train_smiles, train_morgan_stats)
    x_val_hybrid = append_dense_features(x_val_smiles, val_morgan_stats)
    x_test_hybrid = append_dense_features(x_test_smiles, test_morgan_stats)
    x_test_missing_hybrid = append_dense_features(x_test_missing_smiles, missing_morgan_stats)

    hybrid_smiles_model = train_classifier(x_train_hybrid, y_train)
    hybrid_smiles_val_prob = hybrid_smiles_model.predict_proba(x_val_hybrid)[:, 1]
    hybrid_smiles_test_prob = hybrid_smiles_model.predict_proba(x_test_hybrid)[:, 1]
    hybrid_smiles_missing_prob = hybrid_smiles_model.predict_proba(x_test_missing_hybrid)[:, 1]

    x_train_hybrid_plus = sparse.hstack([x_train_hybrid, x_train_morgan], format="csr")
    x_val_hybrid_plus = sparse.hstack([x_val_hybrid, x_val_morgan], format="csr")
    x_test_hybrid_plus = sparse.hstack([x_test_hybrid, x_test_morgan], format="csr")
    x_test_missing_hybrid_plus = sparse.hstack([x_test_missing_hybrid, x_test_missing_morgan], format="csr")

    hybrid_plus_model = train_classifier(x_train_hybrid_plus, y_train)
    hybrid_plus_val_prob = hybrid_plus_model.predict_proba(x_val_hybrid_plus)[:, 1]
    hybrid_plus_test_prob = hybrid_plus_model.predict_proba(x_test_hybrid_plus)[:, 1]
    hybrid_plus_missing_prob = hybrid_plus_model.predict_proba(x_test_missing_hybrid_plus)[:, 1]

    hybrid_blend_val_prob = 0.5 * (hybrid_smiles_val_prob + hybrid_plus_val_prob)
    hybrid_blend_test_prob = 0.5 * (hybrid_smiles_test_prob + hybrid_plus_test_prob)
    hybrid_blend_missing_prob = 0.5 * (hybrid_smiles_missing_prob + hybrid_plus_missing_prob)

    hybrid_pretrained_val_prob = None
    hybrid_pretrained_test_prob = None
    hybrid_pretrained_missing_prob = None
    hybrid_smiles_text_val_prob = None
    hybrid_smiles_text_test_prob = None
    hybrid_smiles_text_missing_prob = None
    retrieval_pretrained_smiles_text_blend_alpha = None
    retrieval_pretrained_smiles_text_val_prob = None
    retrieval_pretrained_smiles_text_test_prob = None
    retrieval_pretrained_smiles_text_missing_prob = None

    if smiles_embedder is not None:
        train_smiles_emb = smiles_embedder.transform(train_df["smiles"])
        val_smiles_emb = smiles_embedder.transform(val_df["smiles"])
        test_smiles_emb = smiles_embedder.transform(test_df["smiles"])
        missing_smiles_emb = smiles_embedder.transform(stressed_test_df["smiles"])

        x_train_hybrid_pretrained = append_dense_features(x_train_hybrid_plus, train_smiles_emb)
        x_val_hybrid_pretrained = append_dense_features(x_val_hybrid_plus, val_smiles_emb)
        x_test_hybrid_pretrained = append_dense_features(x_test_hybrid_plus, test_smiles_emb)
        x_test_missing_hybrid_pretrained = append_dense_features(
            x_test_missing_hybrid_plus,
            missing_smiles_emb,
        )

        hybrid_pretrained_model = train_classifier(x_train_hybrid_pretrained, y_train)
        hybrid_pretrained_val_prob = hybrid_pretrained_model.predict_proba(x_val_hybrid_pretrained)[:, 1]
        hybrid_pretrained_test_prob = hybrid_pretrained_model.predict_proba(x_test_hybrid_pretrained)[:, 1]
        hybrid_pretrained_missing_prob = hybrid_pretrained_model.predict_proba(x_test_missing_hybrid_pretrained)[:, 1]

        if text_embedder is not None:
            train_text_emb = text_embedder.transform(train_df["text"])
            val_text_emb = text_embedder.transform(val_df["text"])
            test_text_emb = text_embedder.transform(test_df["text"])
            missing_text_emb = text_embedder.transform(stressed_test_df["text"])

            x_train_hybrid_smiles_text = append_dense_features(x_train_hybrid_pretrained, train_text_emb)
            x_val_hybrid_smiles_text = append_dense_features(x_val_hybrid_pretrained, val_text_emb)
            x_test_hybrid_smiles_text = append_dense_features(x_test_hybrid_pretrained, test_text_emb)
            x_test_missing_hybrid_smiles_text = append_dense_features(
                x_test_missing_hybrid_pretrained,
                missing_text_emb,
            )

            hybrid_smiles_text_model = train_classifier(x_train_hybrid_smiles_text, y_train)
            hybrid_smiles_text_val_prob = hybrid_smiles_text_model.predict_proba(x_val_hybrid_smiles_text)[:, 1]
            hybrid_smiles_text_test_prob = hybrid_smiles_text_model.predict_proba(x_test_hybrid_smiles_text)[:, 1]
            hybrid_smiles_text_missing_prob = hybrid_smiles_text_model.predict_proba(
                x_test_missing_hybrid_smiles_text
            )[:, 1]

            retrieval_pretrained_smiles_text_blend_alpha = select_blend_weight(
                y_val,
                retrieval_val_prob,
                hybrid_smiles_text_val_prob,
            )
            retrieval_pretrained_smiles_text_val_prob = blend_probabilities(
                retrieval_val_prob,
                hybrid_smiles_text_val_prob,
                retrieval_pretrained_smiles_text_blend_alpha,
            )
            retrieval_pretrained_smiles_text_test_prob = blend_probabilities(
                retrieval_test_prob,
                hybrid_smiles_text_test_prob,
                retrieval_pretrained_smiles_text_blend_alpha,
            )
            retrieval_pretrained_smiles_text_missing_prob = blend_probabilities(
                retrieval_missing_prob,
                hybrid_smiles_text_missing_prob,
                retrieval_pretrained_smiles_text_blend_alpha,
            )

    predictions = pd.DataFrame(
        {
            "sample_id": test_df["sample_id"] if "sample_id" in test_df.columns else [f"row_{i}" for i in range(len(test_df))],
            "target_id": test_df["target_id"] if "target_id" in test_df.columns else "",
            "label": y_test,
            "mask_prob_clean": mask_test_prob,
            "retrieval_prob_clean": retrieval_test_prob,
            "hybrid_blend_avg_prob_clean": hybrid_blend_test_prob,
            "hybrid_plus_pretrained_smiles_prob_clean": hybrid_pretrained_test_prob,
            "hybrid_plus_pretrained_smiles_text_prob_clean": hybrid_smiles_text_test_prob,
            "retrieval_pretrained_smiles_text_tuned_prob_clean": retrieval_pretrained_smiles_text_test_prob,
            "mask_prob_missing": mask_missing_prob,
            "retrieval_prob_missing": retrieval_missing_prob,
            "hybrid_blend_avg_prob_missing": hybrid_blend_missing_prob,
            "hybrid_plus_pretrained_smiles_prob_missing": hybrid_pretrained_missing_prob,
            "hybrid_plus_pretrained_smiles_text_prob_missing": hybrid_smiles_text_missing_prob,
            "retrieval_pretrained_smiles_text_tuned_prob_missing": retrieval_pretrained_smiles_text_missing_prob,
            "text_missing": stressed_test_df["text"].eq(""),
            "sequence_missing": stressed_test_df["sequence"].eq(""),
        }
    )

    models = {
        "mask": {
            "val": evaluate_binary(y_val, mask_val_prob),
            "test_clean": evaluate_binary(y_test, mask_test_prob),
            "test_missing": evaluate_binary(y_test, mask_missing_prob),
        },
        "retrieval": {
            "val": evaluate_binary(y_val, retrieval_val_prob),
            "test_clean": evaluate_binary(y_test, retrieval_test_prob),
            "test_missing": evaluate_binary(y_test, retrieval_missing_prob),
        },
        "hybrid_blend_avg": {
            "val": evaluate_binary(y_val, hybrid_blend_val_prob),
            "test_clean": evaluate_binary(y_test, hybrid_blend_test_prob),
            "test_missing": evaluate_binary(y_test, hybrid_blend_missing_prob),
        },
    }
    if smiles_embedder is not None:
        models["hybrid_plus_pretrained_smiles"] = {
            "val": evaluate_binary(y_val, hybrid_pretrained_val_prob),
            "test_clean": evaluate_binary(y_test, hybrid_pretrained_test_prob),
            "test_missing": evaluate_binary(y_test, hybrid_pretrained_missing_prob),
        }
    if smiles_embedder is not None and text_embedder is not None:
        models["hybrid_plus_pretrained_smiles_text"] = {
            "val": evaluate_binary(y_val, hybrid_smiles_text_val_prob),
            "test_clean": evaluate_binary(y_test, hybrid_smiles_text_test_prob),
            "test_missing": evaluate_binary(y_test, hybrid_smiles_text_missing_prob),
        }
        models["retrieval_pretrained_smiles_text_tuned"] = {
            "val": evaluate_binary(y_val, retrieval_pretrained_smiles_text_val_prob),
            "test_clean": evaluate_binary(y_test, retrieval_pretrained_smiles_text_test_prob),
            "test_missing": evaluate_binary(y_test, retrieval_pretrained_smiles_text_missing_prob),
            "blend_alpha": float(retrieval_pretrained_smiles_text_blend_alpha),
        }

    return {"models": models, "predictions": predictions}


def run_single_experiment(
    dataset: str,
    sample_size: int,
    cache_dir: Path,
    seed: int,
    split_mode: str,
    missing_sequence_prob: float,
    missing_text_prob: float,
    n_neighbors: int = 8,
    smiles_embedder=None,
    text_embedder=None,
    sequence_embedder=None,
    enable_interaction_probe: bool = False,
    interaction_probe_config: dict | None = None,
    retrieval_reference_size: int | None = None,
) -> dict:
    bundle = load_bundle(dataset=dataset, cache_dir=cache_dir, sample_size=sample_size, seed=seed)
    splits = make_splits(bundle.frame, split_mode=split_mode, seed=seed)
    stressed_test_df = inject_missing_modalities(
        splits["test"],
        probs={"sequence": missing_sequence_prob, "text": missing_text_prob},
        seed=seed + 1,
    )
    suite = run_model_suite(
        train_df=splits["train"],
        val_df=splits["val"],
        test_df=splits["test"],
        stressed_test_df=stressed_test_df,
        n_neighbors=n_neighbors,
        smiles_embedder=smiles_embedder,
        text_embedder=text_embedder,
        sequence_embedder=sequence_embedder,
        enable_interaction_probe=enable_interaction_probe,
        interaction_probe_config=interaction_probe_config,
        retrieval_reference_size=retrieval_reference_size,
    )
    metrics = {
        "dataset": dataset,
        "sample_size": int(len(bundle.frame)),
        "target_text_source": bundle.target_text_source,
        "split_mode": split_mode,
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "missing_sequence_prob": missing_sequence_prob,
        "missing_text_prob": missing_text_prob,
        "models": suite["models"],
    }
    metrics["delta_mask_vs_nomask_missing_auroc"] = (
        metrics["models"]["mask"]["test_missing"]["auroc"]
        - metrics["models"]["no_mask"]["test_missing"]["auroc"]
    )
    metrics["delta_retrieval_vs_mask_missing_auroc"] = (
        metrics["models"]["retrieval"]["test_missing"]["auroc"]
        - metrics["models"]["mask"]["test_missing"]["auroc"]
    )
    metrics["delta_retrieval_vs_mask_missing_auprc"] = (
        metrics["models"]["retrieval"]["test_missing"]["auprc"]
        - metrics["models"]["mask"]["test_missing"]["auprc"]
    )
    metrics["predictions"] = suite["predictions"]
    metrics["validation_predictions"] = suite["validation_predictions"]
    metrics["train_preview"] = splits["train"]
    return metrics


def run_single_tdc_official_experiment(
    dataset: str,
    cache_dir: Path,
    seed: int,
    split_method: str,
    missing_sequence_prob: float,
    missing_text_prob: float,
    n_neighbors: int = 8,
    smiles_embedder=None,
    text_embedder=None,
    sequence_embedder=None,
    enable_interaction_probe: bool = False,
    interaction_probe_config: dict | None = None,
    activity_threshold_nm: float = 1000.0,
    inactive_threshold_nm: float | None = None,
    retrieval_reference_size: int | None = None,
) -> dict:
    splits = prepare_public_dti_official_splits(
        dataset_name=dataset,
        cache_dir=cache_dir,
        split_method=split_method,
        seed=seed,
        activity_threshold_nm=activity_threshold_nm,
        inactive_threshold_nm=inactive_threshold_nm,
    )
    stressed_test_df = inject_missing_modalities(
        splits["test"],
        probs={"sequence": missing_sequence_prob, "text": missing_text_prob},
        seed=seed + 1,
    )
    suite = run_model_suite(
        train_df=splits["train"],
        val_df=splits["val"],
        test_df=splits["test"],
        stressed_test_df=stressed_test_df,
        n_neighbors=n_neighbors,
        smiles_embedder=smiles_embedder,
        text_embedder=text_embedder,
        sequence_embedder=sequence_embedder,
        enable_interaction_probe=enable_interaction_probe,
        interaction_probe_config=interaction_probe_config,
        retrieval_reference_size=retrieval_reference_size,
    )
    metrics = {
        "dataset": dataset,
        "sample_size": int(sum(len(frame) for frame in splits.values())),
        "target_text_source": "UniProt",
        "split_mode": split_method,
        "split_source": "tdc_official",
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "missing_sequence_prob": missing_sequence_prob,
        "missing_text_prob": missing_text_prob,
        "models": suite["models"],
    }
    metrics["delta_mask_vs_nomask_missing_auroc"] = (
        metrics["models"]["mask"]["test_missing"]["auroc"]
        - metrics["models"]["no_mask"]["test_missing"]["auroc"]
    )
    metrics["delta_retrieval_vs_mask_missing_auroc"] = (
        metrics["models"]["retrieval"]["test_missing"]["auroc"]
        - metrics["models"]["mask"]["test_missing"]["auroc"]
    )
    metrics["delta_retrieval_vs_mask_missing_auprc"] = (
        metrics["models"]["retrieval"]["test_missing"]["auprc"]
        - metrics["models"]["mask"]["test_missing"]["auprc"]
    )
    metrics["predictions"] = suite["predictions"]
    metrics["validation_predictions"] = suite["validation_predictions"]
    metrics["train_preview"] = splits["train"]
    return metrics

