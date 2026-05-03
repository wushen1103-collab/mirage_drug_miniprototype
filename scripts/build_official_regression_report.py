from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.external_baselines import affinity_to_regression_target
from mirage_mini.reporting import infer_seed_from_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--patterns", nargs="*", default=[])
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--output-dir", default="outputs/official_regression_report")
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


def _summary_stat(series: pd.Series, stat: str) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return float("nan")
    if stat == "mean":
        return float(values.mean())
    if stat == "std":
        return float(values.std(ddof=0))
    if stat == "ci95":
        if len(values) == 1:
            return 0.0
        return float(1.96 * values.std(ddof=0) / np.sqrt(len(values)))
    raise ValueError(f"Unsupported summary stat: {stat}")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred)[0])


def ci(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    order = np.argsort(y_true)
    y_true = y_true[order]
    y_pred = y_pred[order]
    i = len(y_true) - 1
    j = i - 1
    z = 0.0
    score = 0.0
    while i > 0:
        while j >= 0:
            if y_true[i] > y_true[j]:
                z += 1
                diff = y_pred[i] - y_pred[j]
                if diff > 0:
                    score += 1
                elif diff == 0:
                    score += 0.5
            j -= 1
        i -= 1
        j = i - 1
    return float(score / z) if z else float("nan")


def fit_monotonic_bridge(x_val: np.ndarray, y_val: np.ndarray):
    x_val = np.asarray(x_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float)
    valid = np.isfinite(x_val) & np.isfinite(y_val)
    x_val = x_val[valid]
    y_val = y_val[valid]
    if len(x_val) == 0:
        raise ValueError("No finite validation rows available for post-hoc regression.")
    if len(np.unique(x_val)) < 2:
        constant = float(np.mean(y_val))

        def predict(scores: np.ndarray) -> np.ndarray:
            scores = np.asarray(scores, dtype=float)
            return np.full(scores.shape, constant, dtype=float)

        return predict, "constant_mean"

    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(x_val, y_val)
    return model.predict, "isotonic"


def evaluate_bridge(
    *,
    dataset_name: str,
    frame: pd.DataFrame,
    score_column: str,
    predictor,
) -> tuple[np.ndarray, np.ndarray]:
    if score_column not in frame.columns:
        raise KeyError(f"Missing score column: {score_column}")
    working = frame[["label_raw", score_column]].copy()
    working["label_raw"] = pd.to_numeric(working["label_raw"], errors="coerce")
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working = working.dropna()
    if dataset_name.upper().strip() != "KIBA":
        working = working[working["label_raw"].gt(0)].copy()
    if working.empty:
        raise ValueError(f"No usable rows found for score column {score_column}.")
    y_true = affinity_to_regression_target(working["label_raw"].to_numpy(), dataset_name=dataset_name)
    y_pred = predictor(working[score_column].to_numpy(dtype=float))
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def main() -> None:
    args = parse_args()
    outputs_root = REPO_ROOT / args.outputs_root
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = resolve_run_dirs(outputs_root=outputs_root, patterns=args.patterns, explicit_run_dirs=args.run_dirs)
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
        selected_models = detected_models if not args.models else [model for model in detected_models if model in set(args.models)]

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

            y_val = affinity_to_regression_target(validation_frame["label_raw"].to_numpy(), dataset_name=dataset_name)
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
            std_rmse=("rmse", lambda x: _summary_stat(x, "std")),
            ci95_rmse=("rmse", lambda x: _summary_stat(x, "ci95")),
            mean_pearson=("pearson", lambda x: _summary_stat(x, "mean")),
            std_pearson=("pearson", lambda x: _summary_stat(x, "std")),
            ci95_pearson=("pearson", lambda x: _summary_stat(x, "ci95")),
            mean_spearman=("spearman", lambda x: _summary_stat(x, "mean")),
            std_spearman=("spearman", lambda x: _summary_stat(x, "std")),
            ci95_spearman=("spearman", lambda x: _summary_stat(x, "ci95")),
            mean_ci=("ci", lambda x: _summary_stat(x, "mean")),
            std_ci=("ci", lambda x: _summary_stat(x, "std")),
            ci95_ci=("ci", lambda x: _summary_stat(x, "ci95")),
        )
        .reset_index()
        .sort_values(["split_mode", "metric_split", "mean_rmse", "mean_ci"], ascending=[True, True, True, False])
        .reset_index(drop=True)
    )

    per_run.to_csv(output_dir / "official_regression_per_run.csv", index=False)
    summary.to_csv(output_dir / "official_regression_summary.csv", index=False)
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

