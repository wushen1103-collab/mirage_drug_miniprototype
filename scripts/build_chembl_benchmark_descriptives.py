from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import (
    prepare_chembl_assay_subset,
    split_assay_cold,
    split_random,
    split_target_cold,
    split_temporal,
)
from mirage_mini.external_baselines import affinity_to_regression_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", default="outputs/chembl_benchmark_descriptives")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    return parser.parse_args()


def safe_year_min(frame: pd.DataFrame) -> float:
    if "document_year" not in frame.columns or frame["document_year"].dropna().empty:
        return float("nan")
    return float(pd.to_numeric(frame["document_year"], errors="coerce").dropna().min())


def safe_year_max(frame: pd.DataFrame) -> float:
    if "document_year" not in frame.columns or frame["document_year"].dropna().empty:
        return float("nan")
    return float(pd.to_numeric(frame["document_year"], errors="coerce").dropna().max())


def describe_frame(frame: pd.DataFrame, *, split_mode: str | None = None, split_name: str | None = None) -> dict:
    pkd = affinity_to_regression_target(frame["label_raw"].to_numpy(), dataset_name="CHEMBL_ASSAY")
    return {
        "split_mode": split_mode,
        "split_name": split_name,
        "rows": int(len(frame)),
        "positive_rate": float(frame["label"].mean()),
        "unique_drugs": int(frame["drug_id"].nunique()),
        "unique_targets": int(frame["target_id"].nunique()),
        "unique_assays": int(frame["assay_id"].nunique()) if "assay_id" in frame.columns else 0,
        "min_year": safe_year_min(frame),
        "max_year": safe_year_max(frame),
        "mean_label_raw_nm": float(pd.to_numeric(frame["label_raw"], errors="coerce").mean()),
        "median_label_raw_nm": float(pd.to_numeric(frame["label_raw"], errors="coerce").median()),
        "mean_pkd": float(pd.Series(pkd).mean()),
        "median_pkd": float(pd.Series(pkd).median()),
        "missing_text_rate": float(frame["text"].fillna("").eq("").mean()),
        "missing_sequence_rate": float(frame["sequence"].fillna("").eq("").mean()),
    }


def overlap_rows(split_mode: str, frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    for entity in ["drug_id", "target_id", "assay_id"]:
        if entity not in frames["train"].columns:
            continue
        for left_name, right_name in [("train", "val"), ("train", "test"), ("val", "test")]:
            left = set(frames[left_name][entity].dropna().astype(str))
            right = set(frames[right_name][entity].dropna().astype(str))
            union = left | right
            overlap = left & right
            rows.append(
                {
                    "split_mode": split_mode,
                    "entity": entity,
                    "pair": f"{left_name}_vs_{right_name}",
                    "left_unique": int(len(left)),
                    "right_unique": int(len(right)),
                    "overlap_count": int(len(overlap)),
                    "jaccard": float(len(overlap) / len(union)) if union else float("nan"),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    cache_dir = REPO_ROOT / args.cache_dir
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = prepare_chembl_assay_subset(
        cache_dir=cache_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        activity_threshold_nm=args.activity_threshold_nm,
    )

    split_builders = {
        "random": lambda frame: split_random(frame, seed=args.seed),
        "target_cold": lambda frame: split_target_cold(frame, seed=args.seed),
        "assay_cold": lambda frame: split_assay_cold(frame, seed=args.seed),
        "temporal": lambda frame: split_temporal(frame, seed=args.seed),
    }

    overview = {
        "sample_size_requested": args.sample_size,
        "sample_size_realized": int(len(bundle.frame)),
        "seed": args.seed,
        "activity_threshold_nm": args.activity_threshold_nm,
        "target_text_source": bundle.target_text_source,
        "dataset_overview": describe_frame(bundle.frame),
    }

    split_rows: list[dict] = []
    overlap_audit_rows: list[dict] = []
    for split_mode, builder in split_builders.items():
        frames = builder(bundle.frame)
        for split_name, split_frame in frames.items():
            split_rows.append(describe_frame(split_frame, split_mode=split_mode, split_name=split_name))
        overlap_audit_rows.extend(overlap_rows(split_mode=split_mode, frames=frames))

    (output_dir / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    pd.DataFrame(split_rows).to_csv(output_dir / "split_summary.csv", index=False)
    pd.DataFrame(overlap_audit_rows).to_csv(output_dir / "split_overlap_audit.csv", index=False)
    bundle.frame.to_csv(output_dir / "benchmark_frame.csv", index=False)
    print(json.dumps(overview, indent=2))


if __name__ == "__main__":
    main()

