from __future__ import annotations

import pandas as pd
import pytest

from mirage_mini.external_baselines import nm_to_pkd, prepare_external_regression_frame
from mirage_mini.reporting import load_external_run_tables, summarize_external_run_tables


def test_nm_to_pkd_matches_common_affinity_thresholds():
    values = nm_to_pkd([1000.0, 10000.0, 100.0])
    assert list(values.round(6)) == [6.0, 5.0, 7.0]


def test_prepare_external_regression_frame_renames_sequences_and_converts_targets():
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "drug_id": ["d1", "d2"],
            "target_id": ["t1", "t2"],
            "smiles": ["CCO", "CCC"],
            "sequence": ["MKT", "AAAA"],
            "label": [1, 0],
            "label_raw": [1000.0, 10000.0],
        }
    )

    converted = prepare_external_regression_frame(frame)

    assert list(converted.columns) == [
        "sample_id",
        "drug_id",
        "target_id",
        "Drug",
        "Target",
        "Y",
        "binary_label",
        "label_raw_nm",
    ]
    assert converted.loc[0, "Drug"] == "CCO"
    assert converted.loc[0, "Target"] == "MKT"
    assert converted.loc[0, "Y"] == 6.0
    assert converted.loc[1, "Y"] == 5.0


def test_prepare_external_regression_frame_drops_nonpositive_affinities():
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "drug_id": ["d1", "d2"],
            "target_id": ["t1", "t2"],
            "smiles": ["CCO", "CCC"],
            "sequence": ["MKT", "AAAA"],
            "label": [1, 0],
            "label_raw": [0.0, 1000.0],
        }
    )

    converted = prepare_external_regression_frame(frame)

    assert len(converted) == 1
    assert converted.iloc[0]["sample_id"] == "s2"


def test_external_reporting_loads_and_summarizes_metrics(tmp_path):
    outputs_root = tmp_path / "outputs"
    run_a = outputs_root / "balm_projection_cold_target_s42"
    run_b = outputs_root / "balm_projection_cold_target_s43"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "external_metrics.json").write_text(
        """
        {
          "framework": "balm",
          "model": "balm_projection",
          "dataset": "BindingDB_Kd",
          "split_mode": "cold_target",
          "metric_split": "test_clean",
          "seed": 42,
          "metrics": {"auprc": 0.81, "auroc": 0.84, "rmse": 1.2, "pearson": 0.5, "spearman": 0.49, "ci": 0.72}
        }
        """,
        encoding="utf-8",
    )
    (run_b / "external_metrics.json").write_text(
        """
        {
          "framework": "balm",
          "model": "balm_projection",
          "dataset": "BindingDB_Kd",
          "split_mode": "cold_target",
          "metric_split": "test_clean",
          "seed": 43,
          "metrics": {"auprc": 0.83, "auroc": 0.85, "rmse": 1.1, "pearson": 0.52, "spearman": 0.50, "ci": 0.73}
        }
        """,
        encoding="utf-8",
    )

    frame = load_external_run_tables(outputs_root=outputs_root, patterns=["balm_projection_cold_target_s*"])
    summary = summarize_external_run_tables(frame)

    assert len(frame) == 2
    assert list(frame["seed"]) == [42, 43]
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["runs"] == 2
    assert row["mean_auprc"] == pytest.approx(0.82)
    assert row["mean_auroc"] == pytest.approx(0.845)

