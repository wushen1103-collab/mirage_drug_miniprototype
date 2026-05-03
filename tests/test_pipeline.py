from __future__ import annotations

import numpy as np
import pandas as pd
import subprocess
import sys
import json
from pathlib import Path

from mirage_mini.data import (
    _auto_max_chembl_pages,
    _default_uniprot_text_fetcher,
    finalize_activity_frame,
    prepare_public_dti_official_splits,
    split_assay_cold,
    split_random,
    split_target_cold,
    split_temporal,
)
from mirage_mini.experiment import run_model_suite
from mirage_mini.experiment import run_benchmark_shortlist_suite
from mirage_mini.experiment import run_single_tdc_official_experiment
from mirage_mini.features import MultiModalFeaturizer, MorganFingerprintFeaturizer
from mirage_mini.metrics import evaluate_binary
from mirage_mini.model import (
    append_dense_features,
    blend_probabilities,
    build_gate_features,
    fit_gate_model,
    select_best_candidate,
    select_blend_weight,
    train_classifier,
)
from mirage_mini.retrieval import MultiViewRetrievalAugmentor, RetrievalAugmentor, TanimotoRetrievalAugmentor


def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCl", "CCBr"],
            "sequence": ["MKTAA", "MKTAV", "", "GGHHT", "GGHHS", ""],
            "text": [
                "kinase receptor tyrosine",
                "kinase receptor tyrosine",
                "gpcr dopamine receptor",
                "acetyl transferase enzyme",
                "",
                "ion channel protein",
            ],
            "label": [1, 1, 0, 0, 0, 1],
        }
    )


def test_multimodal_features_include_masks():
    frame = _toy_frame()
    features = MultiModalFeaturizer().transform(frame)
    assert features.matrix.shape[0] == len(frame)
    assert features.masks.shape == (len(frame), 3)
    assert features.smiles_matrix.shape[0] == len(frame)
    assert features.sequence_matrix.shape[0] == len(frame)
    assert features.text_matrix.shape[0] == len(frame)
    assert features.masks[2, 1] == 0.0
    assert features.masks[4, 2] == 0.0


def test_retrieval_augmentor_returns_dense_stats():
    frame = _toy_frame()
    x = MultiModalFeaturizer().transform(frame)
    y = frame["label"].to_numpy()
    retrieval = RetrievalAugmentor(n_neighbors=2).fit(x.matrix, y)
    train_stats = retrieval.transform_train(x.matrix)
    assert train_stats.shape == (len(frame), 5)
    assert np.isfinite(train_stats).all()


def test_multiview_retrieval_augmentor_returns_rich_stats():
    frame = _toy_frame()
    x = MultiModalFeaturizer().transform(frame)
    y = frame["label"].to_numpy()
    retrieval = MultiViewRetrievalAugmentor(n_neighbors=2).fit(x, y)
    train_stats = retrieval.transform_train(x)
    query_stats = retrieval.transform(x)
    assert train_stats.shape[0] == len(frame)
    assert query_stats.shape == train_stats.shape
    assert train_stats.shape[1] > 5
    assert np.isfinite(train_stats).all()
    assert np.isfinite(query_stats).all()


def test_morgan_featurizer_returns_binary_sparse_matrix():
    frame = _toy_frame()
    matrix = MorganFingerprintFeaturizer(n_bits=128, radius=2).transform(frame["smiles"])
    assert matrix.shape == (len(frame), 128)
    assert matrix.nnz > 0
    assert set(np.unique(matrix.data)).issubset({1.0})


def test_tanimoto_retrieval_augmentor_returns_dense_stats():
    frame = _toy_frame()
    morgan = MorganFingerprintFeaturizer(n_bits=128, radius=2)
    x = morgan.transform(frame["smiles"])
    y = frame["label"].to_numpy()
    retrieval = TanimotoRetrievalAugmentor(n_neighbors=2).fit(x, y)
    train_stats = retrieval.transform_train(x)
    query_stats = retrieval.transform(x)
    assert train_stats.shape == (len(frame), 6)
    assert query_stats.shape == (len(frame), 6)
    assert np.isfinite(train_stats).all()
    assert np.isfinite(query_stats).all()


def test_retrieval_augmentor_chunked_matches_default():
    frame = _toy_frame()
    x = MultiModalFeaturizer().transform(frame)
    y = frame["label"].to_numpy()
    base = RetrievalAugmentor(n_neighbors=2).fit(x.matrix, y)
    chunked = RetrievalAugmentor(n_neighbors=2, chunk_size=1).fit(x.matrix, y)
    assert np.allclose(base.transform_train(x.matrix), chunked.transform_train(x.matrix))
    assert np.allclose(base.transform(x.matrix), chunked.transform(x.matrix))


def test_multiview_retrieval_augmentor_chunked_matches_default():
    frame = _toy_frame()
    x = MultiModalFeaturizer().transform(frame)
    y = frame["label"].to_numpy()
    base = MultiViewRetrievalAugmentor(n_neighbors=2).fit(x, y)
    chunked = MultiViewRetrievalAugmentor(n_neighbors=2, chunk_size=1).fit(x, y)
    assert np.allclose(base.transform_train(x), chunked.transform_train(x))
    assert np.allclose(base.transform(x), chunked.transform(x))


def test_tanimoto_retrieval_augmentor_chunked_matches_default():
    frame = _toy_frame()
    x = MorganFingerprintFeaturizer(n_bits=128, radius=2).transform(frame["smiles"])
    y = frame["label"].to_numpy()
    base = TanimotoRetrievalAugmentor(n_neighbors=2).fit(x, y)
    chunked = TanimotoRetrievalAugmentor(n_neighbors=2, chunk_size=1).fit(x, y)
    assert np.allclose(base.transform_train(x), chunked.transform_train(x))
    assert np.allclose(base.transform(x), chunked.transform(x))


def test_retrieval_augmentor_parallel_matches_default():
    frame = _toy_frame()
    x = MultiModalFeaturizer().transform(frame)
    y = frame["label"].to_numpy()
    base = RetrievalAugmentor(n_neighbors=2).fit(x.matrix, y)
    parallel = RetrievalAugmentor(n_neighbors=2, n_jobs=2).fit(x.matrix, y)
    assert np.allclose(base.transform_train(x.matrix), parallel.transform_train(x.matrix))
    assert np.allclose(base.transform(x.matrix), parallel.transform(x.matrix))


def test_multiview_retrieval_augmentor_parallel_matches_default():
    frame = _toy_frame()
    x = MultiModalFeaturizer().transform(frame)
    y = frame["label"].to_numpy()
    base = MultiViewRetrievalAugmentor(n_neighbors=2).fit(x, y)
    parallel = MultiViewRetrievalAugmentor(n_neighbors=2, n_jobs=2).fit(x, y)
    assert np.allclose(base.transform_train(x), parallel.transform_train(x))
    assert np.allclose(base.transform(x), parallel.transform(x))


def test_tanimoto_retrieval_augmentor_parallel_matches_default():
    frame = _toy_frame()
    x = MorganFingerprintFeaturizer(n_bits=128, radius=2).transform(frame["smiles"])
    y = frame["label"].to_numpy()
    base = TanimotoRetrievalAugmentor(n_neighbors=2).fit(x, y)
    parallel = TanimotoRetrievalAugmentor(n_neighbors=2, n_jobs=2).fit(x, y)
    assert np.allclose(base.transform_train(x), parallel.transform_train(x))
    assert np.allclose(base.transform(x), parallel.transform(x))


def test_retrieval_augmentor_large_reference_budget_matches_full():
    frame = _toy_frame()
    x = MultiModalFeaturizer().transform(frame)
    y = frame["label"].to_numpy()
    base = RetrievalAugmentor(n_neighbors=2).fit(x.matrix, y)
    banked = RetrievalAugmentor(n_neighbors=2, max_reference_size=64).fit(x.matrix, y)
    assert np.allclose(base.transform_train(x.matrix), banked.transform_train(x.matrix))
    assert np.allclose(base.transform(x.matrix), banked.transform(x.matrix))


def test_multiview_retrieval_augmentor_large_reference_budget_matches_full():
    frame = _toy_frame()
    x = MultiModalFeaturizer().transform(frame)
    y = frame["label"].to_numpy()
    base = MultiViewRetrievalAugmentor(n_neighbors=2).fit(x, y)
    banked = MultiViewRetrievalAugmentor(n_neighbors=2, max_reference_size=64).fit(x, y)
    assert np.allclose(base.transform_train(x), banked.transform_train(x))
    assert np.allclose(base.transform(x), banked.transform(x))


def test_tanimoto_retrieval_augmentor_large_reference_budget_matches_full():
    frame = _toy_frame()
    x = MorganFingerprintFeaturizer(n_bits=128, radius=2).transform(frame["smiles"])
    y = frame["label"].to_numpy()
    base = TanimotoRetrievalAugmentor(n_neighbors=2).fit(x, y)
    banked = TanimotoRetrievalAugmentor(n_neighbors=2, max_reference_size=64).fit(x, y)
    assert np.allclose(base.transform_train(x), banked.transform_train(x))
    assert np.allclose(base.transform(x), banked.transform(x))


def test_end_to_end_training_on_toy_data():
    frame = _toy_frame()
    train = frame.iloc[:4].reset_index(drop=True)
    test = frame.iloc[4:].reset_index(drop=True)
    x_train = MultiModalFeaturizer().transform(train)
    x_test = MultiModalFeaturizer().transform(test)
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    baseline = train_classifier(x_train.matrix, y_train)
    baseline_prob = baseline.predict_proba(x_test.matrix)[:, 1]

    retrieval = RetrievalAugmentor(n_neighbors=2).fit(x_train.matrix, y_train)
    x_train_aug = append_dense_features(x_train.matrix, retrieval.transform_train(x_train.matrix))
    x_test_aug = append_dense_features(x_test.matrix, retrieval.transform(x_test.matrix))
    model = train_classifier(x_train_aug, y_train)
    retrieval_prob = model.predict_proba(x_test_aug)[:, 1]

    baseline_metrics = evaluate_binary(y_test, baseline_prob)
    retrieval_metrics = evaluate_binary(y_test, retrieval_prob)
    assert "auroc" in baseline_metrics
    assert "auprc" in retrieval_metrics


def test_gate_model_produces_probabilities():
    mask_prob = np.array([0.2, 0.8, 0.3, 0.7], dtype=np.float32)
    retrieval_prob = np.array([0.1, 0.9, 0.4, 0.6], dtype=np.float32)
    retrieval_stats = np.array(
        [
            [0.1, 0.0, 0.8, 0.9, 1.0, 0.2],
            [0.9, 0.0, 0.8, 0.9, 1.0, 0.8],
            [0.2, 0.0, 0.7, 0.8, 1.0, 0.3],
            [0.8, 0.0, 0.7, 0.8, 1.0, 0.7],
        ],
        dtype=np.float32,
    )
    masks = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    gate_x = build_gate_features(mask_prob, retrieval_prob, retrieval_stats, masks)
    gate = fit_gate_model(gate_x, np.array([0, 1, 0, 1], dtype=np.int64))
    pred = gate.predict_proba(gate_x)
    assert pred.shape == (4,)
    assert np.all(pred >= 0.0)
    assert np.all(pred <= 1.0)


def test_finalize_activity_frame_creates_binary_labels():
    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "CCC"],
            "sequence": ["AAAA", "BBBB", "CCCC"],
            "text": ["binding assay 1", "binding assay 2", "binding assay 3"],
            "target_id": ["P1", "P2", "P3"],
            "assay_id": ["A1", "A2", "A3"],
            "label_raw": [100.0, 10000.0, 3000.0],
        }
    )
    out = finalize_activity_frame(frame, activity_threshold_nm=1000.0)
    assert len(out) == 2
    assert out["label"].tolist() == [1, 0]


def test_finalize_activity_frame_preserves_optional_text_columns():
    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN"],
            "sequence": ["AAAA", "BBBB"],
            "text": ["binding assay kinase", "binding assay gpcr"],
            "assay_text": ["binding assay", "binding assay"],
            "target_text": ["kinase", "gpcr"],
            "target_id": ["P1", "P2"],
            "assay_id": ["A1", "A2"],
            "label_raw": [100.0, 10000.0],
        }
    )
    out = finalize_activity_frame(frame, activity_threshold_nm=1000.0)
    assert "assay_text" in out.columns
    assert "target_text" in out.columns
    assert out["assay_text"].tolist() == ["binding assay", "binding assay"]
    assert out["target_text"].tolist() == ["kinase", "gpcr"]


def test_auto_max_chembl_pages_scales_for_larger_samples():
    assert _auto_max_chembl_pages(350) == 12
    assert _auto_max_chembl_pages(1000) == 24
    assert _auto_max_chembl_pages(2000) == 48
    assert _auto_max_chembl_pages(4000) == 72


def test_split_target_cold_falls_back_for_tiny_target_sets():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "target_id": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "smiles": ["CCO"] * 6,
            "sequence": ["AAAA"] * 6,
            "text": ["desc"] * 6,
            "label": [1, 0, 1, 0, 1, 0],
            "label_raw": [1, 2, 3, 4, 5, 6],
        }
    )
    splits = split_target_cold(frame, seed=1)
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0


def test_split_assay_cold_holds_out_assays():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(9)],
            "target_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4", "P5"],
            "assay_id": ["A1", "A1", "A2", "A2", "A3", "A3", "A4", "A4", "A5"],
            "smiles": ["CCO"] * 9,
            "sequence": ["AAAA"] * 9,
            "text": ["desc"] * 9,
            "label": [1, 0, 1, 0, 1, 0, 1, 0, 1],
            "label_raw": list(range(9)),
        }
    )
    splits = split_assay_cold(frame, seed=1)
    train_assays = set(splits["train"]["assay_id"])
    val_assays = set(splits["val"]["assay_id"])
    test_assays = set(splits["test"]["assay_id"])
    assert train_assays.isdisjoint(val_assays)
    assert train_assays.isdisjoint(test_assays)
    assert val_assays.isdisjoint(test_assays)


def test_split_temporal_orders_years():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "target_id": [f"P{i // 2}" for i in range(12)],
            "assay_id": [f"A{i // 2}" for i in range(12)],
            "document_year": [2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023],
            "smiles": ["CCO"] * 12,
            "sequence": ["AAAA"] * 12,
            "text": ["desc"] * 12,
            "label": [1, 0] * 6,
            "label_raw": list(range(12)),
        }
    )
    splits = split_temporal(frame, seed=1)
    train_years = set(splits["train"]["document_year"])
    val_years = set(splits["val"]["document_year"])
    test_years = set(splits["test"]["document_year"])
    assert train_years
    assert val_years
    assert test_years
    assert train_years.isdisjoint(val_years)
    assert train_years.isdisjoint(test_years)
    assert val_years.isdisjoint(test_years)
    assert max(train_years) < min(val_years)
    assert max(val_years) < min(test_years)


def test_split_temporal_falls_back_for_tiny_year_sets():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(6)],
            "target_id": ["P1", "P1", "P2", "P2", "P3", "P3"],
            "assay_id": ["A1", "A1", "A2", "A2", "A3", "A3"],
            "document_year": [2020, 2020, 2020, 2021, 2021, 2021],
            "smiles": ["CCO"] * 6,
            "sequence": ["AAAA"] * 6,
            "text": ["desc"] * 6,
            "label": [1, 0, 1, 0, 1, 0],
            "label_raw": list(range(6)),
        }
    )
    splits = split_temporal(frame, seed=7)
    assert len(splits["train"]) > 0
    assert len(splits["val"]) > 0
    assert len(splits["test"]) > 0
    assert splits["train"]["document_year"].max() <= splits["val"]["document_year"].min()
    assert splits["val"]["document_year"].max() <= splits["test"]["document_year"].min()


def test_prepare_public_dti_official_splits_uses_cold_pair_protocol(tmp_path):
    class FakeDTI:
        def __init__(self, df):
            self.df = df
            self.calls = []

        def get_split(self, method="random", seed=42, frac=None, column_name=None):
            self.calls.append(
                {
                    "method": method,
                    "seed": seed,
                    "frac": frac,
                    "column_name": column_name,
                }
            )
            return {
                "train": self.df.iloc[:2].copy(),
                "valid": self.df.iloc[2:3].copy(),
                "test": self.df.iloc[3:].copy(),
            }

    raw = pd.DataFrame(
        {
            "Drug_ID": ["D1", "D2", "D3", "D4"],
            "Drug": ["CCO", "CCN", "CCC", "CCCl"],
            "Target_ID": ["P1", "P2", "P3", "P4"],
            "Target": ["AAAA", "BBBB", "CCCC", "DDDD"],
            "Y": [100.0, 12000.0, 200.0, 15000.0],
        }
    )
    dataset = FakeDTI(raw)

    splits = prepare_public_dti_official_splits(
        dataset_name="BindingDB_Kd",
        cache_dir=tmp_path,
        split_method="cold_pair",
        seed=7,
        dataset_loader=lambda dataset_name, cache_dir: dataset,
        text_fetcher=lambda accessions, cache_path: {str(x): f"{x} description" for x in accessions},
    )

    assert dataset.calls == [
        {
            "method": "cold_split",
            "seed": 7,
            "frac": [0.7, 0.1, 0.2],
            "column_name": ["Drug_ID", "Target_ID"],
        }
    ]
    assert set(splits.keys()) == {"train", "val", "test"}
    assert list(splits["train"].columns) == [
        "sample_id",
        "drug_id",
        "target_id",
        "smiles",
        "sequence",
        "text",
        "label",
        "label_raw",
    ]
    assert splits["train"]["text"].tolist() == ["P1 description", "P2 description"]
    assert splits["train"]["label"].tolist() == [1, 0]
    assert splits["val"]["label"].tolist() == [1]
    assert splits["test"]["label"].tolist() == [0]


def test_prepare_public_dti_official_splits_uses_cold_drug_method(tmp_path):
    class FakeDTI:
        def __init__(self, df):
            self.df = df
            self.calls = []

        def get_split(self, method="random", seed=42, frac=None, column_name=None):
            self.calls.append(
                {
                    "method": method,
                    "seed": seed,
                    "frac": frac,
                    "column_name": column_name,
                }
            )
            return {
                "train": self.df.iloc[:2].copy(),
                "valid": self.df.iloc[2:3].copy(),
                "test": self.df.iloc[3:].copy(),
            }

    raw = pd.DataFrame(
        {
            "Drug_ID": ["D1", "D2", "D3", "D4"],
            "Drug": ["CCO", "CCN", "CCC", "CCCl"],
            "Target_ID": ["P1", "P1", "P2", "P2"],
            "Target": ["AAAA", "AAAA", "BBBB", "BBBB"],
            "Y": [50.0, 15000.0, 500.0, 12000.0],
        }
    )
    dataset = FakeDTI(raw)

    splits = prepare_public_dti_official_splits(
        dataset_name="BindingDB_Kd",
        cache_dir=tmp_path,
        split_method="cold_drug",
        seed=11,
        dataset_loader=lambda dataset_name, cache_dir: dataset,
        text_fetcher=lambda accessions, cache_path: {str(x): str(x) for x in accessions},
    )

    assert dataset.calls == [
        {
            "method": "cold_drug",
            "seed": 11,
            "frac": [0.7, 0.1, 0.2],
            "column_name": None,
        }
    ]
    assert len(splits["train"]) == 2
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1


def test_default_uniprot_text_fetcher_supports_legacy_string_cache(tmp_path):
    cache_path = tmp_path / "bindingdb_kd_uniprot_text_cache.json"
    cache_path.write_text(
        json.dumps(
            {
                "P12345": "legacy kinase description",
                "Q99999": {"text": "structured receptor description", "sequence": "MSEQ"},
            }
        ),
        encoding="utf-8",
    )

    text_map = _default_uniprot_text_fetcher(["P12345", "Q99999"], cache_path=cache_path)

    assert text_map["P12345"] == "legacy kinase description"
    assert text_map["Q99999"] == "structured receptor description"


def test_run_single_tdc_official_experiment_uses_official_loader(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "drug_id": ["D1", "D2"],
            "target_id": ["P1", "P2"],
            "smiles": ["CCO", "CCN"],
            "sequence": ["AAAA", "BBBB"],
            "text": ["desc1", "desc2"],
            "label": [1, 0],
            "label_raw": [100.0, 15000.0],
        }
    )
    calls = {}

    def fake_prepare(**kwargs):
        calls["prepare_kwargs"] = kwargs
        return {
            "train": frame.iloc[:1].copy(),
            "val": frame.iloc[:1].copy(),
            "test": frame.iloc[1:].copy(),
        }

    def fake_run_model_suite(**kwargs):
        calls["suite_kwargs"] = kwargs
        return {
            "models": {
                "no_mask": {"test_missing": {"auroc": 0.50, "auprc": 0.50}},
                "mask": {"test_missing": {"auroc": 0.60, "auprc": 0.60}},
                "retrieval": {"test_missing": {"auroc": 0.70, "auprc": 0.80}},
            },
            "predictions": pd.DataFrame({"sample_id": ["s2"], "label": [0]}),
            "validation_predictions": pd.DataFrame({"sample_id": ["s1"], "label": [1]}),
        }

    monkeypatch.setattr("mirage_mini.experiment.prepare_public_dti_official_splits", fake_prepare)
    monkeypatch.setattr("mirage_mini.experiment.run_model_suite", fake_run_model_suite)

    metrics = run_single_tdc_official_experiment(
        dataset="BindingDB_Kd",
        cache_dir=tmp_path,
        seed=5,
        split_method="cold_target",
        missing_sequence_prob=0.25,
        missing_text_prob=0.5,
        retrieval_reference_size=17,
    )

    assert calls["prepare_kwargs"]["split_method"] == "cold_target"
    assert calls["suite_kwargs"]["train_df"]["sample_id"].tolist() == ["s1"]
    assert calls["suite_kwargs"]["retrieval_reference_size"] == 17
    assert metrics["split_source"] == "tdc_official"
    assert metrics["split_mode"] == "cold_target"
    assert metrics["split_sizes"] == {"train": 1, "val": 1, "test": 1}
    assert metrics["sample_size"] == 3
    assert np.isclose(metrics["delta_mask_vs_nomask_missing_auroc"], 0.10)
    assert np.isclose(metrics["delta_retrieval_vs_mask_missing_auprc"], 0.20)


def test_random_split_preserves_total_rows():
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(10)],
            "target_id": ["P1"] * 10,
            "assay_id": ["A1"] * 10,
            "smiles": ["CCO"] * 10,
            "sequence": ["AAAA"] * 10,
            "text": ["desc"] * 10,
            "label": [1, 0] * 5,
            "label_raw": list(range(10)),
        }
    )
    splits = split_random(frame, seed=7)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == len(frame)


def test_featurizer_can_disable_presence_masks():
    frame = _toy_frame()
    with_masks = MultiModalFeaturizer().transform(frame)
    without_masks = MultiModalFeaturizer(include_masks=False).transform(frame)
    assert with_masks.matrix.shape[1] == without_masks.matrix.shape[1] + 3


def test_run_model_suite_reports_all_ablation_variants():
    frame = _toy_frame()
    train = frame.iloc[:4].reset_index(drop=True)
    val = frame.iloc[4:5].reset_index(drop=True)
    test = frame.iloc[5:].reset_index(drop=True)
    suite = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
    )
    assert {
        "no_mask",
        "mask",
        "retrieval",
        "smiles_only_retrieval",
        "morgan_smiles_retrieval",
        "hybrid_smiles_retrieval",
        "hybrid_plus_morgan_bits",
        "hybrid_blend_avg",
        "multiview_retrieval",
        "gated_retrieval",
    } <= set(suite["models"].keys())
    assert "test_clean" in suite["models"]["retrieval"]


def test_features_module_exposes_cached_transformer_embedder():
    import mirage_mini.features as features

    assert hasattr(features, "CachedTransformerEmbedder")


def test_cached_transformer_embedder_deduplicates_and_reuses_cache(tmp_path):
    from mirage_mini.features import CachedTransformerEmbedder

    class DummyBackend:
        dim = 3

        def __init__(self):
            self.calls = []

        def encode(self, texts, batch_size=32):
            self.calls.append(list(texts))
            rows = []
            for idx, text in enumerate(texts):
                rows.append([float(len(text)), float(idx), 1.0])
            return np.asarray(rows, dtype=np.float32)

    backend = DummyBackend()
    embedder = CachedTransformerEmbedder(
        model_name="dummy/model",
        cache_path=tmp_path / "dummy_cache.pkl",
        backend=backend,
    )
    texts = pd.Series(["CCO", "CCN", "CCO", ""])
    first = embedder.transform(texts)
    second = embedder.transform(pd.Series(["CCN", "CCO", ""]))

    assert first.shape == (4, 3)
    assert second.shape == (3, 3)
    assert len(backend.calls) == 1
    assert sorted(backend.calls[0]) == ["CCN", "CCO"]
    assert np.allclose(first[0], first[2])
    assert np.allclose(second[0], first[1])
    assert np.allclose(first[-1], np.zeros(3, dtype=np.float32))


def test_run_model_suite_reports_pretrained_variants_when_embedder_is_provided():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = _toy_frame()
    train = frame.iloc[:4].reset_index(drop=True)
    val = frame.iloc[4:5].reset_index(drop=True)
    test = frame.iloc[5:].reset_index(drop=True)
    suite = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        sequence_embedder=DummyEmbedder(),
    )
    assert {
        "pretrained_smiles_dense",
        "pretrained_smiles_retrieval",
        "pretrained_text_retrieval",
        "hybrid_plus_pretrained_sequence",
        "hybrid_plus_pretrained_smiles_sequence",
        "hybrid_plus_pretrained_smiles",
        "hybrid_blend_pretrained_tuned",
        "hybrid_pretrained_val_select",
        "hybrid_plus_pretrained_text",
        "hybrid_blend_pretrained_text_tuned",
        "hybrid_plus_pretrained_smiles_text",
        "hybrid_blend_pretrained_smiles_text_tuned",
        "gated_strong_pretrained_tuned",
        "robust_sequence_anchor",
        "retrieval_sequence_bridge",
        "sequence_gate_tuned",
        "fullsuite_val_select",
        "hybrid_plus_pretrained_text_retrieval",
        "hybrid_plus_pretrained_smiles_text_retrieval",
    } <= set(suite["models"].keys())


def test_run_benchmark_shortlist_suite_reports_main_candidates():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = _toy_frame()
    train = frame.iloc[:4].reset_index(drop=True)
    val = frame.iloc[4:5].reset_index(drop=True)
    test = frame.iloc[5:].reset_index(drop=True)
    suite = run_benchmark_shortlist_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
    )
    assert {
        "mask",
        "retrieval",
        "hybrid_blend_avg",
        "hybrid_plus_pretrained_smiles",
        "hybrid_plus_pretrained_smiles_text",
        "retrieval_pretrained_smiles_text_tuned",
    } <= set(suite["models"].keys())
    assert "strong_pretrained_gate" not in suite["models"]
    assert "interaction_probe_smiles_text" not in suite["models"]


def test_run_benchmark_shortlist_suite_reports_tuned_retrieval_pretrained_candidate():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCl", "CCBr", "CCF", "CCC"],
            "sequence": ["MKTAA", "MKTAV", "", "GGHHT", "GGHHS", "", "TTAAA", "TTAAB"],
            "text": [
                "kinase receptor tyrosine",
                "kinase receptor enzyme",
                "gpcr dopamine receptor",
                "acetyl transferase enzyme",
                "",
                "ion channel protein",
                "kinase signaling pathway",
                "membrane transport protein",
            ],
            "label": [1, 1, 0, 0, 0, 1, 1, 0],
        }
    )
    train = frame.iloc[:4].reset_index(drop=True)
    val = frame.iloc[4:6].reset_index(drop=True)
    test = frame.iloc[6:].reset_index(drop=True)
    suite = run_benchmark_shortlist_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
    )

    tuned = suite["models"]["retrieval_pretrained_smiles_text_tuned"]
    assert 0.0 <= tuned["blend_alpha"] <= 1.0
    assert tuned["val"]["auprc"] >= max(
        suite["models"]["retrieval"]["val"]["auprc"],
        suite["models"]["hybrid_plus_pretrained_smiles_text"]["val"]["auprc"],
    )
    assert "retrieval_pretrained_smiles_text_tuned_prob_clean" in suite["predictions"].columns
    assert "retrieval_pretrained_smiles_text_tuned_prob_missing" in suite["predictions"].columns


def test_run_model_suite_reports_tuned_gate_fusion_candidate():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCl", "CCBr", "CCF", "CCC"],
            "sequence": ["MKTAA", "MKTAV", "", "GGHHT", "GGHHS", "", "TTAAA", "TTAAB"],
            "text": [
                "kinase receptor tyrosine",
                "kinase receptor enzyme",
                "gpcr dopamine receptor",
                "acetyl transferase enzyme",
                "",
                "ion channel protein",
                "kinase signaling pathway",
                "membrane transport protein",
            ],
            "label": [1, 1, 0, 0, 0, 1, 1, 0],
        }
    )
    train = frame.iloc[:4].reset_index(drop=True)
    val = frame.iloc[4:6].reset_index(drop=True)
    test = frame.iloc[6:].reset_index(drop=True)
    suite = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        sequence_embedder=DummyEmbedder(),
    )

    tuned = suite["models"]["gated_strong_pretrained_tuned"]
    assert 0.0 <= tuned["blend_alpha"] <= 1.0
    assert tuned["val"]["auprc"] >= max(
        suite["models"]["gated_retrieval"]["val"]["auprc"],
        suite["models"]["strong_pretrained_gate"]["val"]["auprc"],
    )
    assert "gated_strong_pretrained_tuned_prob_clean" in suite["predictions"].columns
    assert "gated_strong_pretrained_tuned_prob_missing" in suite["predictions"].columns


def test_run_model_suite_reports_robust_sequence_anchor_candidate():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCl", "CCBr", "CCF", "CCC"],
            "sequence": ["MKTAA", "MKTAV", "", "GGHHT", "GGHHS", "", "TTAAA", "TTAAB"],
            "text": [
                "kinase receptor tyrosine",
                "kinase receptor enzyme",
                "gpcr dopamine receptor",
                "acetyl transferase enzyme",
                "",
                "ion channel protein",
                "kinase signaling pathway",
                "membrane transport protein",
            ],
            "label": [1, 1, 0, 0, 0, 1, 1, 0],
        }
    )
    train = frame.iloc[:4].reset_index(drop=True)
    val = frame.iloc[4:6].reset_index(drop=True)
    test = frame.iloc[6:].reset_index(drop=True)
    suite = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        sequence_embedder=DummyEmbedder(),
    )

    anchor = suite["models"]["robust_sequence_anchor"]
    assert np.isclose(anchor["blend_alpha"], 0.35)
    assert "robust_sequence_anchor_prob_clean" in suite["predictions"].columns
    assert "robust_sequence_anchor_prob_missing" in suite["predictions"].columns


def test_run_model_suite_reports_sequence_gate_tuned_candidate():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCl", "CCBr", "CCF", "CCC"],
            "sequence": ["MKTAA", "MKTAV", "", "GGHHT", "GGHHS", "", "TTAAA", "TTAAB"],
            "text": [
                "kinase receptor tyrosine",
                "kinase receptor enzyme",
                "gpcr dopamine receptor",
                "acetyl transferase enzyme",
                "",
                "ion channel protein",
                "kinase signaling pathway",
                "membrane transport protein",
            ],
            "label": [1, 1, 0, 0, 0, 1, 1, 0],
        }
    )
    train = frame.iloc[:4].reset_index(drop=True)
    val = frame.iloc[4:6].reset_index(drop=True)
    test = frame.iloc[6:].reset_index(drop=True)
    suite = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        sequence_embedder=DummyEmbedder(),
    )

    seq_gate = suite["models"]["sequence_gate_tuned"]
    assert 0.0 <= seq_gate["val"]["auprc"] <= 1.0
    assert 0.0 <= seq_gate["test_clean"]["auprc"] <= 1.0
    assert "sequence_gate_tuned_prob_clean" in suite["predictions"].columns
    assert "sequence_gate_tuned_prob_missing" in suite["predictions"].columns


def test_run_model_suite_reports_fixed_anchor_candidates():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i:03d}" for i in range(16)],
            "target_id": [f"P{i % 4}" for i in range(16)],
            "assay_id": [f"A{i % 4}" for i in range(16)],
            "smiles": [
                "CCO",
                "CCN",
                "c1ccccc1",
                "CCCl",
                "CCBr",
                "CCF",
                "CCC",
                "CC=O",
                "CCCO",
                "CCCN",
                "c1ccncc1",
                "CCS",
                "CCP",
                "CCI",
                "CC(C)O",
                "CC(C)N",
            ],
            "sequence": [
                "MKTAA",
                "MKTAV",
                "GGHHT",
                "GGHHS",
                "TTAAA",
                "TTAAB",
                "MKLAA",
                "MKLAV",
                "GGQQT",
                "GGQQS",
                "TTPPP",
                "TTPPP",
                "MNNAA",
                "MNNAV",
                "QQHHT",
                "QQHHS",
            ],
            "text": [
                "kinase receptor tyrosine",
                "kinase receptor enzyme",
                "gpcr dopamine receptor",
                "acetyl transferase enzyme",
                "ion channel protein",
                "kinase signaling pathway",
                "membrane transport protein",
                "enzyme inhibitor assay",
                "growth factor receptor",
                "transporter assay",
                "protein interaction assay",
                "lipid kinase receptor",
                "cell cycle kinase",
                "metabolic enzyme binding",
                "allosteric target assay",
                "signal transduction protein",
            ],
            "assay_text": [
                "cell viability kinase assay",
                "enzyme inhibition assay",
                "dopamine receptor functional assay",
                "acetylation biochemical assay",
                "channel patch clamp assay",
                "signaling pathway assay",
                "transport activity assay",
                "binding affinity assay",
                "growth response assay",
                "substrate transport assay",
                "protein complex assay",
                "lipid phosphorylation assay",
                "cell cycle inhibition assay",
                "metabolic turnover assay",
                "allosteric modulation assay",
                "signal pathway binding assay",
            ],
            "label": [1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0],
        }
    )
    train = frame.iloc[:8].reset_index(drop=True)
    val = frame.iloc[8:12].reset_index(drop=True)
    test = frame.iloc[12:].reset_index(drop=True)
    suite = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        sequence_embedder=DummyEmbedder(),
        enable_interaction_probe=True,
        interaction_probe_config={
            "device": "cpu",
            "batch_size": 4,
            "max_epochs": 8,
            "patience": 3,
            "hidden_dim": 16,
            "proj_dim": 8,
            "lr": 5e-3,
            "text_dropout_prob": 0.1,
            "text_field": "assay_text",
        },
    )

    assert np.isclose(suite["models"]["robust_probe_anchor"]["blend_alpha"], 0.45)
    assert np.isclose(suite["models"]["retrieval_sequence_bridge"]["blend_alpha"], 0.45)
    assert "robust_probe_anchor_prob_clean" in suite["predictions"].columns
    assert "robust_probe_anchor_prob_missing" in suite["predictions"].columns
    assert "retrieval_sequence_bridge_prob_clean" in suite["predictions"].columns
    assert "retrieval_sequence_bridge_prob_missing" in suite["predictions"].columns


def test_run_model_suite_reports_fullsuite_val_select_candidate():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = pd.DataFrame(
        {
            "smiles": ["CCO", "CCN", "c1ccccc1", "CC(=O)O", "CCCl", "CCBr", "CCF", "CCC"],
            "sequence": ["MKTAA", "MKTAV", "", "GGHHT", "GGHHS", "", "TTAAA", "TTAAB"],
            "text": [
                "kinase receptor tyrosine",
                "kinase receptor enzyme",
                "gpcr dopamine receptor",
                "acetyl transferase enzyme",
                "",
                "ion channel protein",
                "kinase signaling pathway",
                "membrane transport protein",
            ],
            "label": [1, 1, 0, 0, 0, 1, 1, 0],
        }
    )
    train = frame.iloc[:4].reset_index(drop=True)
    val = frame.iloc[4:6].reset_index(drop=True)
    test = frame.iloc[6:].reset_index(drop=True)
    suite = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        sequence_embedder=DummyEmbedder(),
    )

    selected = suite["models"]["fullsuite_val_select"]
    candidate_names = [
        "hybrid_blend_avg",
        "hybrid_plus_pretrained_smiles_text",
        "hybrid_plus_pretrained_smiles_sequence",
        "gated_retrieval",
        "strong_pretrained_gate",
        "gated_strong_pretrained_tuned",
        "robust_sequence_anchor",
        "robust_probe_anchor",
        "retrieval_sequence_bridge",
        "sequence_gate_tuned",
        "interaction_probe_smiles_text_blend",
    ]
    available = [name for name in candidate_names if name in suite["models"]]
    assert selected["selected_model"] in available
    assert selected["val"]["auprc"] >= max(suite["models"][name]["val"]["auprc"] for name in available)
    assert "fullsuite_val_select_prob_clean" in suite["predictions"].columns
    assert "fullsuite_val_select_prob_missing" in suite["predictions"].columns


def test_run_model_suite_reports_interaction_gate_tuned_candidate():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i:03d}" for i in range(16)],
            "target_id": [f"P{i % 4}" for i in range(16)],
            "assay_id": [f"A{i % 4}" for i in range(16)],
            "smiles": [
                "CCO",
                "CCN",
                "c1ccccc1",
                "CCCl",
                "CCBr",
                "CC(=O)O",
                "CCCO",
                "CCCC",
                "CCO",
                "CCN",
                "c1ccccc1",
                "CCCl",
                "CCBr",
                "CC(=O)O",
                "CCCO",
                "CCCC",
            ],
            "sequence": [
                "MKTAA",
                "MKTAV",
                "GGHHT",
                "GGHHS",
                "AAAAA",
                "BBBBB",
                "CCCCC",
                "DDDDD",
                "MKTAA",
                "MKTAV",
                "GGHHT",
                "GGHHS",
                "AAAAA",
                "BBBBB",
                "CCCCC",
                "DDDDD",
            ],
            "assay_text": [
                "kinase assay alpha",
                "kinase assay beta",
                "gpcr assay gamma",
                "channel assay delta",
            ]
            * 4,
            "target_text": [
                "tyrosine kinase receptor",
                "serine kinase receptor",
                "dopamine gpcr receptor",
                "membrane ion channel",
            ]
            * 4,
            "text": [
                "kinase assay alpha tyrosine kinase receptor",
                "kinase assay beta serine kinase receptor",
                "gpcr assay gamma dopamine gpcr receptor",
                "channel assay delta membrane ion channel",
            ]
            * 4,
            "label": [1, 1, 0, 0] * 4,
        }
    )
    train = frame.iloc[:8].reset_index(drop=True)
    val = frame.iloc[8:12].reset_index(drop=True)
    test = frame.iloc[12:].reset_index(drop=True)
    suite = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        sequence_embedder=DummyEmbedder(),
        enable_interaction_probe=True,
        interaction_probe_config={
            "device": "cpu",
            "batch_size": 4,
            "max_epochs": 8,
            "patience": 3,
            "hidden_dim": 16,
            "proj_dim": 8,
            "lr": 5e-3,
            "text_dropout_prob": 0.1,
            "text_field": "assay_text",
        },
    )

    tuned = suite["models"]["interaction_gate_tuned"]
    assert 0.0 <= tuned["blend_alpha"] <= 1.0
    assert tuned["val"]["auprc"] >= max(
        suite["models"]["gated_strong_pretrained_tuned"]["val"]["auprc"],
        suite["models"]["interaction_probe_smiles_text_blend"]["val"]["auprc"],
    )
    assert "interaction_gate_tuned_prob_clean" in suite["predictions"].columns
    assert "interaction_gate_tuned_prob_missing" in suite["predictions"].columns


def test_run_benchmark_shortlist_suite_large_reference_budget_matches_default():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = _toy_frame()
    train = frame.iloc[:4].reset_index(drop=True)
    val = frame.iloc[4:5].reset_index(drop=True)
    test = frame.iloc[5:].reset_index(drop=True)
    base = run_benchmark_shortlist_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
    )
    banked = run_benchmark_shortlist_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        retrieval_reference_size=64,
    )
    for model_name in base["models"]:
        assert np.isclose(
            base["models"][model_name]["test_clean"]["auroc"],
            banked["models"][model_name]["test_clean"]["auroc"],
            equal_nan=True,
        )


def test_run_model_suite_large_reference_budget_matches_default():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append([float(len(text)), float(text.count("C")), 0.5, 1.0])
            return np.asarray(rows, dtype=np.float32)

    frame = _toy_frame()
    train = frame.iloc[:4].reset_index(drop=True)
    val = frame.iloc[4:5].reset_index(drop=True)
    test = frame.iloc[5:].reset_index(drop=True)
    base = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        sequence_embedder=DummyEmbedder(),
    )
    banked = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        sequence_embedder=DummyEmbedder(),
        retrieval_reference_size=64,
    )
    for model_name in [
        "retrieval",
        "multiview_retrieval",
        "gated_retrieval",
        "strong_pretrained_gate",
    ]:
        assert np.isclose(
            base["models"][model_name]["test_clean"]["auroc"],
            banked["models"][model_name]["test_clean"]["auroc"],
            equal_nan=True,
        )


def test_select_blend_weight_prefers_stronger_signal():
    y_true = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    weak = np.array([0.7, 0.4, 0.6, 0.45, 0.42], dtype=np.float32)
    strong = np.array([0.1, 0.9, 0.2, 0.8, 0.85], dtype=np.float32)

    alpha = select_blend_weight(y_true, weak, strong)
    blended = blend_probabilities(weak, strong, alpha)

    assert alpha >= 0.5
    assert evaluate_binary(y_true, blended)["auprc"] >= max(
        evaluate_binary(y_true, weak)["auprc"],
        evaluate_binary(y_true, strong)["auprc"],
    )


def test_select_best_candidate_prefers_higher_auprc():
    y_true = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    weak = np.array([0.7, 0.4, 0.6, 0.45, 0.42], dtype=np.float32)
    strong = np.array([0.1, 0.9, 0.2, 0.8, 0.85], dtype=np.float32)

    name = select_best_candidate(y_true, {"weak": weak, "strong": strong})

    assert name == "strong"


def test_interaction_probe_training_returns_probability_predictions():
    from mirage_mini.neural_probe import predict_interaction_probe, train_interaction_probe

    rng = np.random.default_rng(0)
    train_smiles = rng.normal(size=(12, 8)).astype(np.float32)
    train_text = rng.normal(size=(12, 6)).astype(np.float32)
    train_masks = rng.integers(0, 2, size=(12, 2)).astype(np.float32)
    y_train = np.array([0, 1] * 6, dtype=np.int64)

    val_smiles = rng.normal(size=(6, 8)).astype(np.float32)
    val_text = rng.normal(size=(6, 6)).astype(np.float32)
    val_masks = rng.integers(0, 2, size=(6, 2)).astype(np.float32)
    y_val = np.array([0, 1, 1, 0, 1, 0], dtype=np.int64)

    test_smiles = rng.normal(size=(4, 8)).astype(np.float32)
    test_text = rng.normal(size=(4, 6)).astype(np.float32)
    test_masks = rng.integers(0, 2, size=(4, 2)).astype(np.float32)

    bundle = train_interaction_probe(
        train_smiles=train_smiles,
        train_text=train_text,
        train_masks=train_masks,
        y_train=y_train,
        val_smiles=val_smiles,
        val_text=val_text,
        val_masks=val_masks,
        y_val=y_val,
        device="cpu",
        batch_size=4,
        max_epochs=8,
        patience=3,
        hidden_dim=16,
        proj_dim=8,
        lr=1e-2,
    )
    pred = predict_interaction_probe(
        bundle=bundle,
        smiles_emb=test_smiles,
        text_emb=test_text,
        masks=test_masks,
        device="cpu",
        batch_size=2,
    )

    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
    assert np.all(pred >= 0.0)
    assert np.all(pred <= 1.0)
    assert bundle.best_epoch >= 0


def test_interaction_probe_training_supports_aux_features():
    from mirage_mini.neural_probe import predict_interaction_probe, train_interaction_probe

    rng = np.random.default_rng(7)
    train_smiles = rng.normal(size=(16, 8)).astype(np.float32)
    train_text = rng.normal(size=(16, 6)).astype(np.float32)
    train_masks = rng.integers(0, 2, size=(16, 3)).astype(np.float32)
    train_aux = rng.normal(size=(16, 5)).astype(np.float32)
    y_train = np.array([0, 1] * 8, dtype=np.int64)

    val_smiles = rng.normal(size=(8, 8)).astype(np.float32)
    val_text = rng.normal(size=(8, 6)).astype(np.float32)
    val_masks = rng.integers(0, 2, size=(8, 3)).astype(np.float32)
    val_aux = rng.normal(size=(8, 5)).astype(np.float32)
    y_val = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)

    test_smiles = rng.normal(size=(5, 8)).astype(np.float32)
    test_text = rng.normal(size=(5, 6)).astype(np.float32)
    test_masks = rng.integers(0, 2, size=(5, 3)).astype(np.float32)
    test_aux = rng.normal(size=(5, 5)).astype(np.float32)

    bundle = train_interaction_probe(
        train_smiles=train_smiles,
        train_text=train_text,
        train_masks=train_masks,
        train_aux=train_aux,
        y_train=y_train,
        val_smiles=val_smiles,
        val_text=val_text,
        val_masks=val_masks,
        val_aux=val_aux,
        y_val=y_val,
        device="cpu",
        batch_size=4,
        max_epochs=8,
        patience=3,
        hidden_dim=16,
        proj_dim=8,
        lr=5e-3,
        text_dropout_prob=0.2,
    )
    pred = predict_interaction_probe(
        bundle=bundle,
        smiles_emb=test_smiles,
        text_emb=test_text,
        masks=test_masks,
        aux=test_aux,
        device="cpu",
        batch_size=2,
    )

    assert pred.shape == (5,)
    assert np.isfinite(pred).all()
    assert np.all(pred >= 0.0)
    assert np.all(pred <= 1.0)
    assert bundle.best_epoch >= 0


def test_run_model_suite_reports_interaction_probe_variants_when_enabled():
    class DummyEmbedder:
        dim = 4

        def transform(self, texts):
            rows = []
            for text in texts:
                text = str(text)
                rows.append(
                    [
                        float(len(text)),
                        float(text.count("C") + text.count("c")),
                        float(text.count("kinase") + text.count("assay")),
                        1.0 if text else 0.0,
                    ]
                )
            return np.asarray(rows, dtype=np.float32)

    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i:03d}" for i in range(16)],
            "target_id": [f"P{i % 4}" for i in range(16)],
            "assay_id": [f"A{i % 4}" for i in range(16)],
            "smiles": [
                "CCO",
                "CCN",
                "c1ccccc1",
                "CCCl",
                "CCBr",
                "CC(=O)O",
                "CCCO",
                "CCCC",
                "CCO",
                "CCN",
                "c1ccccc1",
                "CCCl",
                "CCBr",
                "CC(=O)O",
                "CCCO",
                "CCCC",
            ],
            "sequence": [
                "MKTAA",
                "MKTAV",
                "GGHHT",
                "GGHHS",
                "AAAAA",
                "BBBBB",
                "CCCCC",
                "DDDDD",
                "MKTAA",
                "MKTAV",
                "GGHHT",
                "GGHHS",
                "AAAAA",
                "BBBBB",
                "CCCCC",
                "DDDDD",
            ],
            "assay_text": [
                "kinase assay alpha",
                "kinase assay beta",
                "gpcr assay gamma",
                "channel assay delta",
            ]
            * 4,
            "target_text": [
                "tyrosine kinase receptor",
                "serine kinase receptor",
                "dopamine gpcr receptor",
                "membrane ion channel",
            ]
            * 4,
            "text": [
                "kinase assay alpha tyrosine kinase receptor",
                "kinase assay beta serine kinase receptor",
                "gpcr assay gamma dopamine gpcr receptor",
                "channel assay delta membrane ion channel",
            ]
            * 4,
            "label": [1, 1, 0, 0] * 4,
        }
    )
    train = frame.iloc[:8].reset_index(drop=True)
    val = frame.iloc[8:12].reset_index(drop=True)
    test = frame.iloc[12:].reset_index(drop=True)

    suite = run_model_suite(
        train_df=train,
        val_df=val,
        test_df=test,
        stressed_test_df=test.copy(),
        n_neighbors=2,
        smiles_embedder=DummyEmbedder(),
        text_embedder=DummyEmbedder(),
        enable_interaction_probe=True,
        interaction_probe_config={
            "device": "cpu",
            "batch_size": 4,
            "max_epochs": 8,
            "patience": 3,
            "hidden_dim": 16,
            "proj_dim": 8,
            "lr": 5e-3,
            "text_dropout_prob": 0.1,
            "text_field": "assay_text",
        },
    )

    assert {
        "strong_pretrained_gate",
        "interaction_probe_smiles_text",
        "interaction_probe_smiles_text_blend",
        "interaction_gate_tuned",
    } <= set(suite["models"].keys())
    assert "strong_pretrained_gate_prob_missing" in suite["predictions"].columns
    assert "interaction_probe_smiles_text_prob_missing" in suite["predictions"].columns
    assert "interaction_probe_smiles_text_blend_prob_missing" in suite["predictions"].columns
    assert "interaction_gate_tuned_prob_missing" in suite["predictions"].columns


def test_cli_scripts_show_help_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        repo_root / "scripts" / "run_mini_experiment.py",
        repo_root / "scripts" / "run_experiment_matrix.py",
        repo_root / "scripts" / "run_tdc_shortlist_bank.py",
    ]

    for script in scripts:
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()

