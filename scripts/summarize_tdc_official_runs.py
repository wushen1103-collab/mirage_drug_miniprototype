from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", required=True, help="Glob pattern relative to outputs/, e.g. tdc_bindingdb_kd_cold_target_seqgatepilot_s*_threads64")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--metric-split", default="test_clean", choices=["val", "test_clean", "test_missing"])
    return parser.parse_args()


def _load_metrics(metrics_path: Path, metric_split: str) -> list[dict]:
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = []
    for model_name, model_metrics in payload["models"].items():
        split_metrics = model_metrics.get(metric_split)
        if not isinstance(split_metrics, dict):
            continue
        row = {
            "run_dir": metrics_path.parent.name,
            "seed": payload.get("seed"),
            "dataset": payload.get("dataset"),
            "split_mode": payload.get("split_mode"),
            "split_source": payload.get("split_source"),
            "model": model_name,
            "metric_split": metric_split,
            "auprc": split_metrics.get("auprc"),
            "auroc": split_metrics.get("auroc"),
            "ece": split_metrics.get("ece"),
            "risk_at_80_coverage": split_metrics.get("risk_at_80_coverage"),
        }
        if "blend_alpha" in model_metrics:
            row["blend_alpha"] = model_metrics["blend_alpha"]
        if "selected_model" in model_metrics:
            row["selected_model"] = model_metrics["selected_model"]
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = repo_root / "outputs"
    run_dirs = sorted(outputs_root.glob(args.pattern))
    if not run_dirs:
        raise SystemExit(f"No run directories matched pattern: {args.pattern}")

    rows: list[dict] = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        rows.extend(_load_metrics(metrics_path, metric_split=args.metric_split))
    if not rows:
        raise SystemExit("Matched run directories, but found no readable metrics rows.")

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["dataset", "split_mode", "metric_split", "model"], dropna=False)
        .agg(
            runs=("run_dir", "count"),
            mean_auprc=("auprc", "mean"),
            mean_auroc=("auroc", "mean"),
            mean_ece=("ece", "mean"),
            mean_risk80=("risk_at_80_coverage", "mean"),
        )
        .reset_index()
        .sort_values(["split_mode", "mean_auprc", "mean_auroc"], ascending=[True, False, False])
    )

    if args.output_csv:
        summary.to_csv(args.output_csv, index=False)
    if args.output_json:
        Path(args.output_json).write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

