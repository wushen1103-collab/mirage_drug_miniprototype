from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLUMNS = ["auroc", "auprc", "rmse", "pearson", "spearman", "ci"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/reports/external_baselines")
    return parser.parse_args()


def _load_external_metric(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None
    metrics = payload.get("metrics")
    required = ["framework", "model", "dataset", "split_mode", "seed"]
    if not isinstance(metrics, dict) or any(payload.get(key) is None for key in required):
        return None
    return {
        "run_dir": path.parent.name,
        "framework": payload.get("framework"),
        "model": payload.get("model"),
        "dataset": payload.get("dataset"),
        "split_mode": payload.get("split_mode"),
        "seed": payload.get("seed"),
        "metric_split": payload.get("metric_split", "test_clean"),
        **{metric: metrics.get(metric) for metric in METRIC_COLUMNS},
    }


def load_external_metrics(outputs_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    skipped: list[dict] = []
    for metrics_path in sorted(outputs_root.glob("**/external_metrics.json")):
        row = _load_external_metric(metrics_path)
        if row is None:
            skipped.append({"path": str(metrics_path), "reason": "malformed_or_incomplete"})
            continue
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(skipped)


def _summary_stat(values: pd.Series, stat: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    if stat == "mean":
        return float(numeric.mean())
    if stat == "std":
        return float(numeric.std(ddof=0)) if len(numeric) > 1 else 0.0
    if stat == "ci95":
        return float(1.96 * numeric.std(ddof=0) / np.sqrt(len(numeric))) if len(numeric) > 1 else 0.0
    raise ValueError(f"Unsupported stat: {stat}")


def summarize_external_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    keys = ["dataset", "split_mode", "metric_split", "framework", "model"]
    rows: list[dict] = []
    for key_values, group in metrics.groupby(keys, dropna=False):
        row = dict(zip(keys, key_values))
        row["runs"] = int(len(group))
        seeds = sorted(set(pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int).tolist()))
        row["seeds"] = ",".join(str(seed) for seed in seeds)
        for metric in METRIC_COLUMNS:
            row[f"mean_{metric}"] = _summary_stat(group[metric], "mean")
            row[f"std_{metric}"] = _summary_stat(group[metric], "std")
            row[f"ci95_{metric}"] = _summary_stat(group[metric], "ci95")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def best_by_condition(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[pd.Series] = []
    for _, group in summary.groupby(["dataset", "split_mode", "metric_split"], dropna=False):
        ranked = group.sort_values(["mean_auprc", "mean_auroc"], ascending=[False, False])
        rows.append(ranked.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def coverage_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for key_values, group in metrics.groupby(["dataset", "split_mode", "framework", "model"], dropna=False):
        row = dict(zip(["dataset", "split_mode", "framework", "model"], key_values))
        row["runs"] = int(len(group))
        seeds = sorted(set(pd.to_numeric(group["seed"], errors="coerce").dropna().astype(int).tolist()))
        row["seeds"] = ",".join(str(seed) for seed in seeds)
        row["metric_splits"] = ",".join(sorted(group["metric_split"].dropna().astype(str).unique()))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "split_mode", "framework", "model"]).reset_index(drop=True)


def build_external_baseline_report(*, outputs_root: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics, skipped = load_external_metrics(outputs_root)
    summary = summarize_external_metrics(metrics)
    best = best_by_condition(summary)
    coverage = coverage_table(metrics)

    metrics.to_csv(output_dir / "external_metrics_long.csv", index=False)
    summary.to_csv(output_dir / "external_metrics_summary.csv", index=False)
    best.to_csv(output_dir / "external_best_by_condition.csv", index=False)
    coverage.to_csv(output_dir / "external_coverage.csv", index=False)
    skipped.to_csv(output_dir / "external_metrics_skipped.csv", index=False)

    files_seen = len(metrics) + len(skipped)
    report = {
        "files_seen": int(files_seen),
        "rows_written": int(len(metrics)),
        "files_skipped": int(len(skipped)),
        "datasets": sorted(metrics["dataset"].dropna().astype(str).unique().tolist()) if not metrics.empty else [],
        "frameworks": sorted(metrics["framework"].dropna().astype(str).unique().tolist()) if not metrics.empty else [],
        "output_dir": str(output_dir),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    report = build_external_baseline_report(
        outputs_root=(repo_root / args.outputs_root).resolve(),
        output_dir=(repo_root / args.output_dir).resolve(),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
