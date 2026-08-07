from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "audit_revision_data_and_results.py"
    spec = importlib.util.spec_from_file_location("audit_revision_data_and_results", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(sample_ids, drugs, targets, assays, labels):
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "drug_id": drugs,
            "target_id": targets,
            "assay_id": assays,
            "document_year": [2020] * len(sample_ids),
            "smiles": ["CCO"] * len(sample_ids),
            "sequence": ["AAAA"] * len(sample_ids),
            "assay_text": ["binding"] * len(sample_ids),
            "target_text": ["target"] * len(sample_ids),
            "text": ["binding [SEP] target"] * len(sample_ids),
            "label": labels,
        }
    )


def test_revision_audit_flags_capped_fetch_and_writes_traceability(tmp_path):
    module = _load_script()
    benchmark_dir = tmp_path / "outputs" / "benchmark"
    split_dir = benchmark_dir / "splits_seed42"
    eval_dir = tmp_path / "outputs" / "eval"
    output_dir = tmp_path / "outputs" / "audit"
    split_dir.mkdir(parents=True)
    eval_dir.mkdir(parents=True)

    train = _frame(["s0", "s1"], ["d0", "d1"], ["t0", "t0"], ["a0", "a0"], [0, 1])
    val = _frame(["s2"], ["d2"], ["t1"], ["a1"], [0])
    test = _frame(["s3"], ["d3"], ["t2"], ["a2"], [1])
    benchmark = pd.concat([train, val, test], ignore_index=True)
    benchmark.to_csv(benchmark_dir / "benchmark_frame.csv", index=False)
    pd.DataFrame(
        {
            "standard_type": ["IC50", "IC50"],
            "year": [2020, 2020],
            "offset": [0, 500],
            "page_count": [500, 500],
            "total_count": [5000, 5000],
            "error": ["", ""],
        }
    ).to_csv(benchmark_dir / "fetch_summary.csv", index=False)
    for split_mode in ("target_cold", "assay_cold"):
        train.to_csv(split_dir / f"{split_mode}_train.csv", index=False)
        val.to_csv(split_dir / f"{split_mode}_val.csv", index=False)
        test.to_csv(split_dir / f"{split_mode}_test.csv", index=False)

    run_dir = eval_dir / "target_cold_s42"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"dataset": "CHEMBL_ASSAY", "split_mode": "target_cold", "seed": 42}),
        encoding="utf-8",
    )
    pd.DataFrame({"sample_id": ["s3"], "label": [1], "prob": [0.8]}).to_csv(
        run_dir / "predictions.csv", index=False
    )

    report = module.audit_revision_inputs(
        benchmark_dir=benchmark_dir,
        eval_dir=eval_dir,
        output_dir=output_dir,
    )

    assert report["status"] == "fail"
    assert report["incomplete_fetch_groups"] == 1
    assert report["split_violations"] == []
    assert report["result_runs"] == 1
    assert (output_dir / "data_version.csv").exists()
    assert (output_dir / "run_registry.csv").exists()
    coverage = pd.read_csv(output_dir / "fetch_coverage_audit.csv")
    assert coverage.loc[0, "coverage_fraction"] == 0.2
