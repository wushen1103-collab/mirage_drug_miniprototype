from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/test/wsk/mirage_drug_miniprototype")
INPUT = ROOT / "outputs" / "revision_e22_pretrained_frozen_strict_20260804"
OUTPUT = ROOT / "outputs" / "revision_e22_pretrained_frozen_strict_20260804_summary"
OUTPUT.mkdir(parents=True, exist_ok=True)


def ci95(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    mean = float(arr.mean())
    se = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    return mean - 1.96 * se, mean + 1.96 * se


def main() -> None:
    rows = []
    for metrics_path in sorted(INPUT.glob("*/external_metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        meta = payload.get("run_meta", {})
        rows.append(
            {
                "framework": payload.get("framework"),
                "model": payload.get("model"),
                "split": payload.get("split_mode"),
                "seed": payload.get("seed"),
                "metric_split": payload.get("metric_split"),
                "auprc": metrics.get("auprc"),
                "auroc": metrics.get("auroc"),
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "pearson": metrics.get("pearson"),
                "spearman": metrics.get("spearman"),
                "ci": metrics.get("ci"),
                "train_size": meta.get("train_size"),
                "val_size": meta.get("val_size"),
                "test_size": meta.get("test_size"),
                "params": meta.get("parameter_count"),
                "feature_dim": meta.get("feature_dim"),
                "best_alpha": meta.get("best_alpha"),
                "encode_seconds": meta.get("encode_seconds"),
                "train_seconds": meta.get("train_seconds"),
                "inference_seconds_per_1000": meta.get("inference_seconds_per_1000"),
                "peak_gpu_memory_mb": meta.get("peak_gpu_memory_mb"),
                "gpu_name": meta.get("gpu_name"),
                "run_dir": str(metrics_path.parent),
            }
        )
    per_run = pd.DataFrame(rows)
    per_run.to_csv(OUTPUT / "pretrained_frozen_strict_per_run.csv", index=False)
    if per_run.empty:
        print("No completed pretrained frozen strict runs found.")
        return

    metric_cols = [
        "auprc",
        "auroc",
        "rmse",
        "mae",
        "pearson",
        "spearman",
        "ci",
        "encode_seconds",
        "train_seconds",
        "inference_seconds_per_1000",
        "peak_gpu_memory_mb",
    ]
    summary_rows = []
    for split, group in per_run.groupby("split", sort=True):
        out = {
            "framework": "pretrained_frozen",
            "model": "frozen_esm2_chemberta_ridge",
            "split": split,
            "runs": int(len(group)),
            "seeds": ",".join(str(int(x)) for x in sorted(group["seed"].dropna().astype(int).unique())),
            "metric_split": ",".join(sorted(set(group["metric_split"].dropna().astype(str)))),
            "train_size_mean": float(group["train_size"].mean()),
            "val_size_mean": float(group["val_size"].mean()),
            "test_size_mean": float(group["test_size"].mean()),
            "params_mean": float(group["params"].mean()),
            "feature_dim_mean": float(group["feature_dim"].mean()),
            "best_alpha_mode": str(group["best_alpha"].mode().iloc[0]) if not group["best_alpha"].dropna().empty else "",
        }
        for metric in metric_cols:
            vals = group[metric].dropna().astype(float).tolist()
            out[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            out[f"{metric}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            lo, hi = ci95(vals)
            out[f"{metric}_ci_low"] = lo
            out[f"{metric}_ci_high"] = hi
        summary_rows.append(out)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "pretrained_frozen_strict_summary_by_split.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

