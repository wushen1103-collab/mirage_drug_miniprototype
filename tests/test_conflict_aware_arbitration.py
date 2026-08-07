from __future__ import annotations

import numpy as np
import pandas as pd

from mirage_mini.experiment import _fit_conflict_aware_arbitration, run_model_suite


def test_conflict_aware_arbitration_emits_bounded_predictions_and_diagnostics():
    y_train = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    y_val = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    current_val = np.asarray([0.1, 0.8, 0.2, 0.7, 0.4, 0.9, 0.3, 0.8, 0.2, 0.9, 0.1, 0.8])
    retrieval_val = np.asarray([0.7, 0.3, 0.6, 0.4, 0.2, 0.8, 0.8, 0.2, 0.1, 0.7, 0.5, 0.4])
    current_test = np.asarray([0.15, 0.75, 0.35, 0.85])
    retrieval_test = np.asarray([0.75, 0.25, 0.55, 0.45])
    current_missing = np.asarray([0.25, 0.65, 0.45, 0.75])
    retrieval_missing = np.asarray([0.65, 0.35, 0.55, 0.35])
    val_stats = np.zeros((len(y_val), 6), dtype=np.float32)
    test_stats = np.zeros((len(current_test), 6), dtype=np.float32)
    missing_stats = np.zeros((len(current_test), 6), dtype=np.float32)
    val_masks = np.ones((len(y_val), 3), dtype=np.float32)
    test_masks = np.ones((len(current_test), 3), dtype=np.float32)
    missing_masks = np.asarray([[1, 1, 0], [1, 0, 1], [1, 1, 1], [1, 0, 0]], dtype=np.float32)

    output = _fit_conflict_aware_arbitration(
        y_train=y_train,
        y_val=y_val,
        current_val=current_val,
        retrieval_val=retrieval_val,
        current_test=current_test,
        retrieval_test=retrieval_test,
        current_missing=current_missing,
        retrieval_missing=retrieval_missing,
        val_stats=val_stats,
        test_stats=test_stats,
        missing_stats=missing_stats,
        val_masks=val_masks,
        test_masks=test_masks,
        missing_masks=missing_masks,
        seed=42,
    )

    for split_name, expected_rows in [("val", len(y_val)), ("test", len(current_test)), ("missing", len(current_test))]:
        split = output[split_name]
        for name in ["full", "no_gate", "no_probe", "no_anchor", "gamma", "reliability", "gated"]:
            values = np.asarray(split[name])
            assert len(values) == expected_rows
            assert np.isfinite(values).all()
            assert ((values >= 0.0) & (values <= 1.0)).all()

    assert output["anchor"] == 0.5


class _TinyEmbedder:
    def transform(self, values):
        values = [str(value) for value in values]
        return np.asarray(
            [[len(value), sum(ord(char) for char in value) % 17, value.count("C")] for value in values],
            dtype=np.float32,
        )


def _frame(prefix: str, labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [f"{prefix}_{index}" for index in range(len(labels))],
            "drug_id": [f"d{index % 4}" for index in range(len(labels))],
            "target_id": [f"t{index % 3}" for index in range(len(labels))],
            "assay_id": [f"a{index % 2}" for index in range(len(labels))],
            "document_year": [2020 + index % 3 for index in range(len(labels))],
            "smiles": ["CCO" if label else "NCC" for label in labels],
            "sequence": ["MKTAA" if label else "GGLVV" for label in labels],
            "text": ["active assay" if label else "inactive assay" for label in labels],
            "label": labels,
            "label_raw": [100.0 if label else 10000.0 for label in labels],
        }
    )


def test_full_mirage_path_emits_fixed_model_and_component_ablation_outputs():
    train = _frame("train", [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    val = _frame("val", [0, 1, 0, 1, 0, 1, 0, 1])
    test = _frame("test", [0, 1, 0, 1, 0, 1, 0, 1])
    stressed = test.copy()
    stressed.loc[[1, 3], "text"] = ""
    stressed.loc[[2, 6], "sequence"] = ""
    embedder = _TinyEmbedder()

    suite = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=stressed,
        n_neighbors=2,
        smiles_embedder=embedder,
        text_embedder=embedder,
        sequence_embedder=embedder,
        enable_interaction_probe=False,
    )

    for name in ["mirage_full", "mirage_w_o_gate", "mirage_w_o_probe", "mirage_w_o_anchor"]:
        assert name in suite["models"]
        assert f"{name}_prob_clean" in suite["predictions"]
        assert f"{name}_prob_missing" in suite["predictions"]
    assert "mirage_gate_weight_clean" in suite["predictions"]
    assert "mirage_reliability_clean" in suite["predictions"]
