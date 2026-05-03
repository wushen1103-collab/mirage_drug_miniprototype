from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_conflict_analysis_writes_bin_level_fusion_evidence(tmp_path):
    module = _load_script("build_conflict_analysis")
    outputs_root = tmp_path / "outputs"
    run_dir = outputs_root / "tdc_kiba_cold_drug_fullsuite_s42_vpred"
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps({"dataset": "KIBA", "split_mode": "cold_drug", "seed": 42}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(8)],
            "label": [1, 0, 1, 0, 1, 0, 1, 0],
            "hybrid_plus_pretrained_smiles_text_prob_clean": [0.90, 0.10, 0.88, 0.12, 0.20, 0.80, 0.25, 0.75],
            "hybrid_plus_pretrained_smiles_text_retrieval_prob_clean": [0.88, 0.12, 0.82, 0.18, 0.85, 0.15, 0.80, 0.20],
            "interaction_gate_tuned_prob_clean": [0.91, 0.09, 0.86, 0.14, 0.78, 0.22, 0.75, 0.25],
        }
    ).to_csv(run_dir / "predictions.csv", index=False)

    report = module.build_conflict_report(
        outputs_root=outputs_root,
        output_dir=tmp_path / "report",
        patterns=["tdc_*_vpred"],
        current_model="hybrid_plus_pretrained_smiles_text",
        retrieval_model="hybrid_plus_pretrained_smiles_text_retrieval",
        fusion_models=["interaction_gate_tuned"],
        metric_splits=["test_clean"],
        n_bins=2,
    )

    assert report["runs_analyzed"] == 1
    per_run = pd.read_csv(tmp_path / "report" / "conflict_per_run.csv")
    assert set(per_run["model"]) == {
        "hybrid_plus_pretrained_smiles_text",
        "hybrid_plus_pretrained_smiles_text_retrieval",
        "interaction_gate_tuned",
    }
    assert set(per_run["conflict_bin"]) == {1, 2}
    high_conflict = per_run[
        (per_run["model"] == "interaction_gate_tuned") & (per_run["conflict_bin"] == 2)
    ].iloc[0]
    assert high_conflict["brier_delta_vs_current"] > 0
    assert (tmp_path / "report" / "conflict_summary.csv").exists()
    assert (tmp_path / "report" / "report.json").exists()


def test_chembl_case_studies_merge_external_predictions_and_seed_gaps(tmp_path):
    module = _load_script("build_chembl_case_studies")
    outputs_root = tmp_path / "outputs"
    benchmark_path = tmp_path / "benchmark_frame.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2"],
            "drug_id": ["d0", "d1", "d2"],
            "target_id": ["t0", "t1", "t2"],
            "assay_id": ["a0", "a1", "a2"],
            "document_year": [2001, 2002, 2003],
            "smiles": ["CCO", "CCN", "CCC"],
            "sequence": ["AAAA", "BBBB", "CCCC"],
            "assay_text": ["binding assay", "inhibition assay", "activity assay"],
            "text": ["binding assay target", "inhibition assay target", "activity assay target"],
            "target_text": ["target zero", "target one", "target two"],
            "label": [1, 1, 0],
            "label_raw": [10.0, 50.0, 100000.0],
        }
    ).to_csv(benchmark_path, index=False)

    for framework, predictions in {
        "deepdta": [7.8, 5.1, 7.0],
        "mgraphdta": [7.6, 7.4, 4.5],
    }.items():
        run_dir = outputs_root / f"external_chembl_{framework}_temporal_s42"
        run_dir.mkdir(parents=True)
        (run_dir / "external_metrics.json").write_text(
            json.dumps(
                {
                    "framework": framework,
                    "model": framework,
                    "dataset": "CHEMBL_ASSAY",
                    "split_mode": "temporal",
                    "seed": 42,
                    "metric_split": "test",
                    "metrics": {"auprc": 0.8, "auroc": 0.6, "rmse": 1.2, "ci": 0.55},
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "sample_id": ["s0", "s1", "s2"],
                "drug_id": ["d0", "d1", "d2"],
                "target_id": ["t0", "t1", "t2"],
                "binary_label": [1, 1, 0],
                "label_raw_nm": [10.0, 50.0, 100000.0],
                "pkd_label": [8.0, 7.3, 4.0],
                "pkd_prediction": predictions,
            }
        ).to_csv(run_dir / "test_prediction_with_binary.csv", index=False)

    report = module.build_chembl_case_studies(
        outputs_root=outputs_root,
        benchmark_frame=benchmark_path,
        output_dir=tmp_path / "case_report",
        splits=["temporal", "assay_cold"],
        frameworks=["deepdta", "mgraphdta"],
        seeds=[42, 43],
        top_n=2,
    )

    assert report["prediction_rows"] == 6
    cases = pd.read_csv(tmp_path / "case_report" / "chembl_case_study_candidates.csv")
    assert {"large_regression_error", "cross_model_disagreement"}.issubset(set(cases["case_type"]))
    gaps = pd.read_csv(tmp_path / "case_report" / "chembl_external_seed_gaps.csv")
    assert gaps["is_present"].eq(False).any()
    notes = (tmp_path / "case_report" / "chembl_case_study_notes.md").read_text(encoding="utf-8")
    assert "temporal" in notes
    assert "s1" in notes or "s2" in notes


def test_external_baseline_report_flattens_metrics_and_marks_best(tmp_path):
    module = _load_script("build_external_baseline_report")
    outputs_root = tmp_path / "outputs"
    for run_name, auprc, seed in [
        ("external_deepdta_cold_drug_s42", 0.70, 42),
        ("external_deepdta_cold_drug_s43", 0.74, 43),
        ("external_graphdta_cold_drug_s42", 0.72, 42),
        ("external_graphdta_cold_drug_s42_rerun", 0.71, 42),
    ]:
        run_dir = outputs_root / run_name
        run_dir.mkdir(parents=True)
        (run_dir / "external_metrics.json").write_text(
            json.dumps(
                {
                    "framework": run_name.split("_")[1],
                    "model": run_name.split("_")[1],
                    "dataset": "KIBA",
                    "split_mode": "cold_drug",
                    "seed": seed,
                    "metric_split": "test_clean",
                    "metrics": {"auprc": auprc, "auroc": auprc + 0.1, "rmse": 1.0},
                }
            ),
            encoding="utf-8",
        )
    bad_dir = outputs_root / "external_bad_run"
    bad_dir.mkdir(parents=True)
    (bad_dir / "external_metrics.json").write_text("{}", encoding="utf-8")

    report = module.build_external_baseline_report(
        outputs_root=outputs_root,
        output_dir=tmp_path / "external_report",
    )

    assert report["files_seen"] == 5
    assert report["rows_written"] == 4
    assert report["files_skipped"] == 1
    summary = pd.read_csv(tmp_path / "external_report" / "external_metrics_summary.csv")
    deepdta = summary[summary["framework"] == "deepdta"].iloc[0]
    assert deepdta["runs"] == 2
    assert deepdta["mean_auprc"] == 0.72
    graphdta = summary[summary["framework"] == "graphdta"].iloc[0]
    assert graphdta["runs"] == 2
    assert graphdta["seeds"] == "42"
    best = pd.read_csv(tmp_path / "external_report" / "external_best_by_condition.csv")
    assert best.iloc[0]["framework"] == "deepdta"
