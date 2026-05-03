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

from mirage_mini.metrics import calibration_table, evaluate_binary, risk_coverage_table
from mirage_mini.reporting import infer_seed_from_run_dir, summarize_official_run_tables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--patterns", nargs="*", default=[])
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--output-dir", default="outputs/reliability_report")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def resolve_run_dirs(outputs_root: Path, patterns: list[str], explicit_run_dirs: list[str]) -> list[Path]:
    run_dirs: list[Path] = []
    run_dirs.extend(Path(path).resolve() for path in explicit_run_dirs)
    for pattern in patterns:
        run_dirs.extend(path.resolve() for path in outputs_root.glob(pattern))
    unique = sorted({path: path for path in run_dirs}.values(), key=lambda path: path.name)
    if not unique:
        raise ValueError("No run directories were provided or matched.")
    return unique


def detect_models(validation_frame: pd.DataFrame, prediction_frame: pd.DataFrame) -> list[str]:
    models = set()
    for column in validation_frame.columns:
        if column.endswith("_prob"):
            models.add(column[: -len("_prob")])
    for column in prediction_frame.columns:
        if column.endswith("_prob_clean"):
            models.add(column[: -len("_prob_clean")])
        if column.endswith("_prob_missing"):
            models.add(column[: -len("_prob_missing")])
    return sorted(models)


def append_split_outputs(
    *,
    frame: pd.DataFrame,
    prob_column: str,
    metric_split: str,
    bins: int,
    run_payload: dict,
    summary_rows: list[dict],
    calibration_rows: list[dict],
    risk_rows: list[dict],
) -> None:
    if prob_column not in frame.columns:
        return
    working = frame[["label", prob_column]].copy()
    working["label"] = pd.to_numeric(working["label"], errors="coerce")
    working[prob_column] = pd.to_numeric(working[prob_column], errors="coerce")
    working = working.dropna()
    if working.empty:
        return

    y_true = working["label"].to_numpy(dtype=int)
    prob = working[prob_column].to_numpy(dtype=float)
    metrics = evaluate_binary(y_true, prob)
    summary_rows.append(
        {
            **run_payload,
            "metric_split": metric_split,
            "n": int(len(working)),
            "positive_rate": float(working["label"].mean()),
            **metrics,
        }
    )

    calibration = calibration_table(y_true, prob, bins=bins)
    if not calibration.empty:
        calibration_rows.extend(
            {
                **run_payload,
                "metric_split": metric_split,
                **row,
            }
            for row in calibration.to_dict(orient="records")
        )

    risk_curve = risk_coverage_table(y_true, prob)
    if not risk_curve.empty:
        risk_rows.extend(
            {
                **run_payload,
                "metric_split": metric_split,
                **row,
            }
            for row in risk_curve.to_dict(orient="records")
        )


def main() -> None:
    args = parse_args()
    outputs_root = REPO_ROOT / args.outputs_root
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = resolve_run_dirs(outputs_root=outputs_root, patterns=args.patterns, explicit_run_dirs=args.run_dirs)
    summary_rows: list[dict] = []
    calibration_rows: list[dict] = []
    risk_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "predictions.csv"
        validation_path = run_dir / "validation_predictions.csv"
        if not (metrics_path.exists() and predictions_path.exists()):
            continue

        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        predictions = pd.read_csv(predictions_path)
        validation_predictions = pd.read_csv(validation_path) if validation_path.exists() else pd.DataFrame()
        detected_models = detect_models(validation_predictions, predictions)
        selected_models = detected_models if not args.models else [model for model in detected_models if model in set(args.models)]

        manifest_rows.append(
            {
                "run_dir": run_dir.name,
                "dataset": metrics_payload.get("dataset"),
                "split_mode": metrics_payload.get("split_mode"),
                "seed": metrics_payload.get("seed", infer_seed_from_run_dir(run_dir.name)),
                "models_detected": ",".join(detected_models),
                "models_selected": ",".join(selected_models),
            }
        )

        for model in selected_models:
            run_payload = {
                "run_dir": run_dir.name,
                "seed": metrics_payload.get("seed", infer_seed_from_run_dir(run_dir.name)),
                "dataset": metrics_payload.get("dataset"),
                "split_mode": metrics_payload.get("split_mode"),
                "model": model,
            }
            append_split_outputs(
                frame=validation_predictions,
                prob_column=f"{model}_prob",
                metric_split="val",
                bins=args.bins,
                run_payload=run_payload,
                summary_rows=summary_rows,
                calibration_rows=calibration_rows,
                risk_rows=risk_rows,
            )
            append_split_outputs(
                frame=predictions,
                prob_column=f"{model}_prob_clean",
                metric_split="test_clean",
                bins=args.bins,
                run_payload=run_payload,
                summary_rows=summary_rows,
                calibration_rows=calibration_rows,
                risk_rows=risk_rows,
            )
            append_split_outputs(
                frame=predictions,
                prob_column=f"{model}_prob_missing",
                metric_split="test_missing",
                bins=args.bins,
                run_payload=run_payload,
                summary_rows=summary_rows,
                calibration_rows=calibration_rows,
                risk_rows=risk_rows,
            )

    if not summary_rows:
        raise ValueError("No reliability rows were generated from the provided runs.")

    per_run = pd.DataFrame(summary_rows)
    grouped = summarize_official_run_tables(
        per_run[
            [
                "run_dir",
                "seed",
                "dataset",
                "split_mode",
                "metric_split",
                "model",
                "auprc",
                "auroc",
                "ece",
                "risk_at_80_coverage",
            ]
        ]
    )
    per_run.to_csv(output_dir / "reliability_per_run.csv", index=False)
    grouped.to_csv(output_dir / "reliability_summary.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration_bins.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(output_dir / "risk_coverage_curve.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(output_dir / "run_manifest.csv", index=False)

    report = {
        "runs_analyzed": int(per_run["run_dir"].nunique()),
        "rows_written": int(len(per_run)),
        "models": sorted(per_run["model"].unique().tolist()),
        "metric_splits": sorted(per_run["metric_split"].unique().tolist()),
        "output_dir": str(output_dir),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

