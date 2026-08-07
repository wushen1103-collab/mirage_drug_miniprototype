from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_official_regression_report import (
    REPO_ROOT,
    _summary_stat,
    ci,
    detect_models,
    evaluate_bridge,
    fit_monotonic_bridge,
    pearson,
    resolve_run_dirs,
    rmse,
    spearman,
)
from mirage_mini.external_baselines import affinity_to_regression_target
from mirage_mini.reporting import infer_seed_from_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--patterns", nargs="*", default=[])
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--output-dir", default="outputs/chembl_assay_v2_regression_bridge_report")
    parser.add_argument("--models", nargs="*", default=None)
    return parser.parse_args()


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def main() -> None:
    args = parse_args()
    outputs_root = REPO_ROOT / args.outputs_root
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = resolve_run_dirs(
        outputs_root=outputs_root,
        patterns=args.patterns,
        explicit_run_dirs=args.run_dirs,
    )
    rows: list[dict] = []
    manifest_rows: list[dict] = []

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "predictions.csv"
        validation_path = run_dir / "validation_predictions.csv"
        if not (metrics_path.exists() and predictions_path.exists() and validation_path.exists()):
            continue

        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        dataset_name = str(metrics_payload.get("dataset"))
        predictions = pd.read_csv(predictions_path)
        validation_predictions = pd.read_csv(validation_path)
        detected_models = detect_models(validation_predictions, predictions)
        selected_models = (
            detected_models
            if not args.models
            else [model for model in detected_models if model in set(args.models)]
        )

        manifest_rows.append(
            {
                "run_dir": run_dir.name,
                "dataset": dataset_name,
                "split_mode": metrics_payload.get("split_mode"),
                "seed": metrics_payload.get("seed", infer_seed_from_run_dir(run_dir.name)),
                "models_detected": ",".join(detected_models),
                "models_selected": ",".join(selected_models),
            }
        )

        for model in selected_models:
            validation_col = f"{model}_prob"
            if validation_col not in validation_predictions.columns:
                continue
            validation_frame = validation_predictions[["label_raw", validation_col]].copy()
            validation_frame["label_raw"] = pd.to_numeric(validation_frame["label_raw"], errors="coerce")
            validation_frame[validation_col] = pd.to_numeric(validation_frame[validation_col], errors="coerce")
            validation_frame = validation_frame.dropna()
            if dataset_name.upper().strip() != "KIBA":
                validation_frame = validation_frame[validation_frame["label_raw"].gt(0)].copy()
            if validation_frame.empty:
                continue

            y_val = affinity_to_regression_target(
                validation_frame["label_raw"].to_numpy(),
                dataset_name=dataset_name,
            )
            predictor, bridge_method = fit_monotonic_bridge(
                validation_frame[validation_col].to_numpy(dtype=float),
                np.asarray(y_val, dtype=float),
            )

            for metric_split, score_column in [
                ("test_clean", f"{model}_prob_clean"),
                ("test_missing", f"{model}_prob_missing"),
            ]:
                if score_column not in predictions.columns:
                    continue
                y_true, y_pred = evaluate_bridge(
                    dataset_name=dataset_name,
                    frame=predictions,
                    score_column=score_column,
                    predictor=predictor,
                )
                rows.append(
                    {
                        "run_dir": run_dir.name,
                        "seed": metrics_payload.get("seed", infer_seed_from_run_dir(run_dir.name)),
                        "dataset": dataset_name,
                        "split_mode": metrics_payload.get("split_mode"),
                        "metric_split": metric_split,
                        "model": model,
                        "bridge_method": bridge_method,
                        "target_scale": "raw" if dataset_name.upper().strip() == "KIBA" else "pkd",
                        "n": int(len(y_true)),
                        "rmse": rmse(y_true, y_pred),
                        "mae": mae(y_true, y_pred),
                        "pearson": pearson(y_true, y_pred),
                        "spearman": spearman(y_true, y_pred),
                        "ci": ci(y_true, y_pred),
                    }
                )

    if not rows:
        raise ValueError("No regression bridge rows were generated from the provided runs.")

    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby(
            ["dataset", "split_mode", "metric_split", "model", "bridge_method", "target_scale"],
            dropna=False,
        )
        .agg(
            runs=("run_dir", "count"),
            mean_rmse=("rmse", lambda x: _summary_stat(x, "mean")),
            ci95_rmse=("rmse", lambda x: _summary_stat(x, "ci95")),
            mean_mae=("mae", lambda x: _summary_stat(x, "mean")),
            ci95_mae=("mae", lambda x: _summary_stat(x, "ci95")),
            mean_pearson=("pearson", lambda x: _summary_stat(x, "mean")),
            ci95_pearson=("pearson", lambda x: _summary_stat(x, "ci95")),
            mean_spearman=("spearman", lambda x: _summary_stat(x, "mean")),
            ci95_spearman=("spearman", lambda x: _summary_stat(x, "ci95")),
            mean_ci=("ci", lambda x: _summary_stat(x, "mean")),
            ci95_ci=("ci", lambda x: _summary_stat(x, "ci95")),
        )
        .reset_index()
        .sort_values(["split_mode", "metric_split", "mean_rmse", "mean_ci"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )

    per_run.to_csv(output_dir / "regression_bridge_per_run.csv", index=False)
    summary.to_csv(output_dir / "regression_bridge_summary.csv", index=False)
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

