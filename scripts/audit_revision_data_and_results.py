from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTITY_COLUMNS = ("drug_id", "target_id", "assay_id")
SPLIT_NAMES = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the benchmark version, split integrity, fetch coverage, and result provenance."
    )
    parser.add_argument(
        "--benchmark-dir",
        default="outputs/chembl_assay_v2_20260505",
        help="Directory containing benchmark_frame.csv and splits_seed*/.",
    )
    parser.add_argument(
        "--eval-dir",
        default="outputs/chembl_assay_v2_eval_20260505",
        help="Directory containing per-run metrics and predictions.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/revision_e00_audit_20260729",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def describe_frame(frame: pd.DataFrame, *, source: str) -> dict:
    row = {
        "source": source,
        "rows": int(len(frame)),
        "positive_rate": float(pd.to_numeric(frame["label"], errors="coerce").mean())
        if "label" in frame
        else np.nan,
        "unique_drugs": int(frame["drug_id"].nunique()) if "drug_id" in frame else 0,
        "unique_targets": int(frame["target_id"].nunique()) if "target_id" in frame else 0,
        "unique_assays": int(frame["assay_id"].nunique()) if "assay_id" in frame else 0,
        "duplicate_sample_ids": int(frame["sample_id"].duplicated().sum()) if "sample_id" in frame else 0,
        "duplicate_drug_target_assay": int(
            frame.duplicated(["drug_id", "target_id", "assay_id"]).sum()
        )
        if all(column in frame for column in ENTITY_COLUMNS)
        else 0,
    }
    years = pd.to_numeric(frame.get("document_year"), errors="coerce")
    row["min_year"] = int(years.min()) if years is not None and years.notna().any() else np.nan
    row["max_year"] = int(years.max()) if years is not None and years.notna().any() else np.nan
    for column in ("smiles", "sequence", "assay_text", "target_text", "text"):
        if column in frame:
            values = frame[column].fillna("").astype(str).str.strip()
            row[f"missing_{column}_rate"] = float(values.eq("").mean())
    return row


def split_overlap_rows(split_mode: str, frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    columns = [*ENTITY_COLUMNS, "drug_target_pair"]
    for column in columns:
        if column == "drug_target_pair":
            values = {
                name: set(frame["drug_id"].astype(str) + "::" + frame["target_id"].astype(str))
                for name, frame in frames.items()
            }
        elif all(column in frame for frame in frames.values()):
            values = {name: set(frame[column].dropna().astype(str)) for name, frame in frames.items()}
        else:
            continue
        for left_name, right_name in (("train", "val"), ("train", "test"), ("val", "test")):
            left = values[left_name]
            right = values[right_name]
            overlap = left & right
            union = left | right
            rows.append(
                {
                    "split_mode": split_mode,
                    "entity": column,
                    "pair": f"{left_name}_vs_{right_name}",
                    "left_unique": len(left),
                    "right_unique": len(right),
                    "overlap_count": len(overlap),
                    "jaccard": len(overlap) / len(union) if union else np.nan,
                }
            )
    return rows


def audit_fetch_coverage(fetch_summary: pd.DataFrame) -> pd.DataFrame:
    required = {"standard_type", "year", "page_count", "total_count"}
    missing = required - set(fetch_summary.columns)
    if missing:
        raise ValueError(f"fetch_summary.csv is missing columns: {sorted(missing)}")
    grouped = (
        fetch_summary.groupby(["standard_type", "year"], as_index=False)
        .agg(
            fetched_rows=("page_count", "sum"),
            query_total=("total_count", "max"),
            pages_fetched=("page_count", "size"),
            fetch_errors=("error", lambda values: int(pd.Series(values).fillna("").astype(str).str.len().gt(0).sum())),
        )
        .sort_values(["year", "standard_type"])
    )
    grouped["coverage_fraction"] = grouped["fetched_rows"] / grouped["query_total"].replace(0, np.nan)
    grouped["fetch_complete"] = grouped["fetched_rows"] >= grouped["query_total"]
    return grouped


def discover_split_frames(split_dir: Path) -> tuple[list[dict], list[dict]]:
    split_rows: list[dict] = []
    overlap_rows: list[dict] = []
    split_modes = sorted({path.name.rsplit("_", 1)[0] for path in split_dir.glob("*_train.csv")})
    for split_mode in split_modes:
        frames: dict[str, pd.DataFrame] = {}
        for split_name in SPLIT_NAMES:
            path = split_dir / f"{split_mode}_{split_name}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frames[split_name] = frame
            split_rows.append(describe_frame(frame, source=f"{split_mode}/{split_name}"))
        overlap_rows.extend(split_overlap_rows(split_mode, frames))
    return split_rows, overlap_rows


def iter_metric_records(eval_dir: Path) -> Iterable[dict]:
    for metrics_path in sorted(eval_dir.glob("*/metrics.json")):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            yield {"run_dir": metrics_path.parent.name, "parse_error": repr(exc)}
            continue
        predictions_path = metrics_path.parent / "predictions.csv"
        row = {
            "run_dir": metrics_path.parent.name,
            "metrics_path": display_path(metrics_path),
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_present": predictions_path.exists(),
            "predictions_sha256": sha256_file(predictions_path) if predictions_path.exists() else "",
            "predictions_rows": int(sum(1 for _ in predictions_path.open("r", encoding="utf-8")) - 1)
            if predictions_path.exists()
            else 0,
        }
        for key in ("dataset", "split_mode", "seed", "metric_split", "model"):
            row[key] = payload.get(key)
        yield row


def audit_revision_inputs(
    *, benchmark_dir: Path, eval_dir: Path, output_dir: Path, strict: bool = False
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = benchmark_dir / "benchmark_frame.csv"
    fetch_path = benchmark_dir / "fetch_summary.csv"
    split_dirs = sorted(benchmark_dir.glob("splits_seed*"))
    if not benchmark_path.exists() or not fetch_path.exists() or not split_dirs:
        raise FileNotFoundError("Benchmark directory lacks benchmark_frame.csv, fetch_summary.csv, or splits_seed*/")

    benchmark = pd.read_csv(benchmark_path)
    benchmark_overview = pd.DataFrame([describe_frame(benchmark, source="full_benchmark")])
    fetch_coverage = audit_fetch_coverage(pd.read_csv(fetch_path))
    split_rows, overlap_rows = discover_split_frames(split_dirs[0])
    split_overview = pd.DataFrame(split_rows)
    overlap_audit = pd.DataFrame(overlap_rows)
    result_registry = pd.DataFrame(iter_metric_records(eval_dir))

    files_to_hash = [benchmark_path, fetch_path]
    for optional_name in ("overview.json", "split_summary.csv", "split_overlap_audit.csv"):
        optional_path = benchmark_dir / optional_name
        if optional_path.exists():
            files_to_hash.append(optional_path)
    data_version = pd.DataFrame(
        [
            {
                "path": display_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files_to_hash
        ]
    )

    cold_expectations = {
        "target_cold": "target_id",
        "assay_cold": "assay_id",
    }
    split_violations: list[dict] = []
    for split_mode, entity in cold_expectations.items():
        selected = overlap_audit[
            (overlap_audit["split_mode"] == split_mode)
            & (overlap_audit["entity"] == entity)
            & overlap_audit["pair"].isin(["train_vs_val", "train_vs_test"])
        ]
        for row in selected.to_dict("records"):
            if int(row["overlap_count"]) > 0:
                split_violations.append(row)

    report = {
        "benchmark_dir": display_path(benchmark_dir),
        "eval_dir": display_path(eval_dir),
        "benchmark_rows": int(len(benchmark)),
        "unique_drugs": int(benchmark["drug_id"].nunique()),
        "unique_targets": int(benchmark["target_id"].nunique()),
        "unique_assays": int(benchmark["assay_id"].nunique()),
        "fetch_groups": int(len(fetch_coverage)),
        "incomplete_fetch_groups": int((~fetch_coverage["fetch_complete"]).sum()),
        "minimum_fetch_coverage": float(fetch_coverage["coverage_fraction"].min()),
        "split_violations": split_violations,
        "result_runs": int(len(result_registry)),
        "all_prediction_files_present": bool(result_registry.get("predictions_present", pd.Series(dtype=bool)).all()),
        "status": "fail" if split_violations or (~fetch_coverage["fetch_complete"]).any() else "pass",
    }

    benchmark_overview.to_csv(output_dir / "benchmark_construction_audit.csv", index=False)
    fetch_coverage.to_csv(output_dir / "fetch_coverage_audit.csv", index=False)
    split_overview.to_csv(output_dir / "split_entity_audit.csv", index=False)
    overlap_audit.to_csv(output_dir / "split_overlap_audit_recomputed.csv", index=False)
    result_registry.to_csv(output_dir / "run_registry.csv", index=False)
    data_version.to_csv(output_dir / "data_version.csv", index=False)
    (output_dir / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if strict and report["status"] != "pass":
        raise RuntimeError(json.dumps(report, indent=2))
    return report


def main() -> None:
    args = parse_args()
    report = audit_revision_inputs(
        benchmark_dir=(REPO_ROOT / args.benchmark_dir).resolve(),
        eval_dir=(REPO_ROOT / args.eval_dir).resolve(),
        output_dir=(REPO_ROOT / args.output_dir).resolve(),
        strict=args.strict,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
