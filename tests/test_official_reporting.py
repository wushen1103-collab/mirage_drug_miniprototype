from __future__ import annotations

import json

import pandas as pd

from mirage_mini.reporting import (
    compute_selection_instability,
    infer_seed_from_run_dir,
    load_official_run_tables,
    summarize_official_run_tables,
)


def _write_metrics(run_dir, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")


def test_infer_seed_from_run_dir_parses_seed_suffix():
    assert infer_seed_from_run_dir("tdc_bindingdb_kd_cold_target_anchorfamily_s42_threads64") == 42
    assert infer_seed_from_run_dir("plain_run_without_seed") is None


def test_load_official_run_tables_reads_metrics_and_selector_metadata(tmp_path):
    outputs_root = tmp_path / "outputs"
    _write_metrics(
        outputs_root / "tdc_bindingdb_kd_cold_target_anchorfamily_s42_threads64",
        {
            "dataset": "BindingDB_Kd",
            "split_mode": "cold_target",
            "models": {
                "robust_probe_anchor": {
                    "test_clean": {
                        "auprc": 0.84,
                        "auroc": 0.88,
                        "ece": 0.04,
                        "risk_at_80_coverage": 0.14,
                    }
                },
                "fullsuite_val_select": {
                    "selected_model": "robust_probe_anchor",
                    "test_clean": {
                        "auprc": 0.83,
                        "auroc": 0.87,
                        "ece": 0.05,
                        "risk_at_80_coverage": 0.15,
                    },
                },
            },
        },
    )

    frame = load_official_run_tables(
        outputs_root=outputs_root,
        patterns=["tdc_bindingdb_kd_cold_target_anchorfamily_s*_threads64"],
        metric_split="test_clean",
    )

    assert set(frame["model"]) == {"robust_probe_anchor", "fullsuite_val_select"}
    assert set(frame["seed"]) == {42}
    selector_row = frame.loc[frame["model"] == "fullsuite_val_select"].iloc[0]
    assert selector_row["selected_model"] == "robust_probe_anchor"


def test_summarize_official_run_tables_groups_runs_by_model():
    frame = pd.DataFrame(
        [
            {
                "run_dir": "run_s42",
                "seed": 42,
                "dataset": "BindingDB_Kd",
                "split_mode": "cold_target",
                "metric_split": "test_clean",
                "model": "robust_probe_anchor",
                "auprc": 0.84,
                "auroc": 0.88,
                "ece": 0.04,
                "risk_at_80_coverage": 0.14,
            },
            {
                "run_dir": "run_s43",
                "seed": 43,
                "dataset": "BindingDB_Kd",
                "split_mode": "cold_target",
                "metric_split": "test_clean",
                "model": "robust_probe_anchor",
                "auprc": 0.86,
                "auroc": 0.89,
                "ece": 0.05,
                "risk_at_80_coverage": 0.15,
            },
        ]
    )

    summary = summarize_official_run_tables(frame)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["runs"] == 2
    assert row["mean_auprc"] == 0.85
    assert row["mean_auroc"] == 0.885


def test_compute_selection_instability_reports_match_and_gap():
    frame = pd.DataFrame(
        [
            {
                "run_dir": "run_s42",
                "seed": 42,
                "dataset": "BindingDB_Kd",
                "split_mode": "cold_target",
                "metric_split": "test_clean",
                "model": "robust_probe_anchor",
                "auprc": 0.84,
                "auroc": 0.88,
                "ece": 0.04,
                "risk_at_80_coverage": 0.14,
                "selected_model": None,
            },
            {
                "run_dir": "run_s42",
                "seed": 42,
                "dataset": "BindingDB_Kd",
                "split_mode": "cold_target",
                "metric_split": "test_clean",
                "model": "interaction_gate_tuned",
                "auprc": 0.85,
                "auroc": 0.879,
                "ece": 0.05,
                "risk_at_80_coverage": 0.145,
                "selected_model": None,
            },
            {
                "run_dir": "run_s42",
                "seed": 42,
                "dataset": "BindingDB_Kd",
                "split_mode": "cold_target",
                "metric_split": "test_clean",
                "model": "fullsuite_val_select",
                "auprc": 0.83,
                "auroc": 0.87,
                "ece": 0.05,
                "risk_at_80_coverage": 0.15,
                "selected_model": "robust_probe_anchor",
            },
            {
                "run_dir": "run_s43",
                "seed": 43,
                "dataset": "BindingDB_Kd",
                "split_mode": "cold_target",
                "metric_split": "test_clean",
                "model": "robust_probe_anchor",
                "auprc": 0.86,
                "auroc": 0.89,
                "ece": 0.04,
                "risk_at_80_coverage": 0.14,
                "selected_model": None,
            },
            {
                "run_dir": "run_s43",
                "seed": 43,
                "dataset": "BindingDB_Kd",
                "split_mode": "cold_target",
                "metric_split": "test_clean",
                "model": "interaction_gate_tuned",
                "auprc": 0.85,
                "auroc": 0.88,
                "ece": 0.05,
                "risk_at_80_coverage": 0.15,
                "selected_model": None,
            },
            {
                "run_dir": "run_s43",
                "seed": 43,
                "dataset": "BindingDB_Kd",
                "split_mode": "cold_target",
                "metric_split": "test_clean",
                "model": "fullsuite_val_select",
                "auprc": 0.86,
                "auroc": 0.89,
                "ece": 0.05,
                "risk_at_80_coverage": 0.14,
                "selected_model": "robust_probe_anchor",
            },
        ]
    )

    instability = compute_selection_instability(frame, selector_model="fullsuite_val_select")

    assert list(instability["selected_matches_test_best"]) == [False, True]
    assert list(instability["best_test_model"]) == ["interaction_gate_tuned", "robust_probe_anchor"]
    assert list(instability["test_auprc_gap"]) == [0.01, 0.0]

