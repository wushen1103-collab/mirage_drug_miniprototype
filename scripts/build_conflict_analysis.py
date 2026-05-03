from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.metrics import evaluate_binary
from mirage_mini.reporting import infer_seed_from_run_dir


DEFAULT_CURRENT_MODEL = "hybrid_plus_pretrained_smiles_text"
DEFAULT_RETRIEVAL_MODEL = "hybrid_plus_pretrained_smiles_text_retrieval"
DEFAULT_FUSION_MODELS = [
    "interaction_gate_tuned",
    "robust_probe_anchor",
    "retrieval_sequence_bridge",
    "fullsuite_val_select",
    "gated_strong_pretrained_tuned",
]
DEFAULT_METRIC_SPLITS = ["val", "test_clean", "test_missing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--patterns", nargs="*", default=["tdc_*_vpred"])
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--output-dir", default="outputs/reports_20260419/conflict_analysis")
    parser.add_argument("--current-model", default=DEFAULT_CURRENT_MODEL)
    parser.add_argument("--retrieval-model", default=DEFAULT_RETRIEVAL_MODEL)
    parser.add_argument("--fusion-models", nargs="*", default=DEFAULT_FUSION_MODELS)
    parser.add_argument("--metric-splits", nargs="*", default=DEFAULT_METRIC_SPLITS)
    parser.add_argument("--n-bins", type=int, default=4)
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


def probability_column(model: str, metric_split: str) -> str:
    if metric_split == "val":
        return f"{model}_prob"
    if metric_split in {"test_clean", "test_missing"}:
        suffix = metric_split.removeprefix("test_")
        return f"{model}_prob_{suffix}"
    raise ValueError(f"Unsupported metric split: {metric_split}")


def prediction_path_for_split(run_dir: Path, metric_split: str) -> Path:
    if metric_split == "val":
        return run_dir / "validation_predictions.csv"
    return run_dir / "predictions.csv"


def _read_metrics(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8-sig"))


def _conflict_bins(conflict: pd.Series, n_bins: int) -> pd.Series:
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if conflict.empty:
        return pd.Series(dtype="int64", index=conflict.index)
    bins = min(n_bins, len(conflict))
    if bins == 1:
        return pd.Series(1, index=conflict.index, dtype="int64")
    ranked = conflict.rank(method="first")
    return pd.qcut(ranked, q=bins, labels=False).astype("int64") + 1


def _safe_model_family(model: str, current_model: str, retrieval_model: str) -> str:
    if model == current_model:
        return "current_sample"
    if model == retrieval_model:
        return "retrieval_neighborhood"
    return "fusion_or_gate"


def _metric_row(
    *,
    frame: pd.DataFrame,
    prob_column: str,
    current_column: str,
    run_payload: dict,
    model: str,
    model_family: str,
    conflict_bin: int,
) -> dict:
    y_true = frame["label"].to_numpy(dtype=int)
    prob = frame[prob_column].to_numpy(dtype=float)
    current_prob = frame[current_column].to_numpy(dtype=float)
    metrics = evaluate_binary(y_true, prob)
    brier = float(np.mean((y_true - prob) ** 2))
    current_brier = float(np.mean((y_true - current_prob) ** 2))
    accuracy = float(((prob >= 0.5).astype(int) == y_true).mean())
    return {
        **run_payload,
        "model": model,
        "model_family": model_family,
        "conflict_bin": int(conflict_bin),
        "n": int(len(frame)),
        "positive_rate": float(frame["label"].mean()),
        "mean_conflict": float(frame["conflict"].mean()),
        "median_conflict": float(frame["conflict"].median()),
        "mean_current_prob": float(frame[current_column].mean()),
        "mean_model_prob": float(frame[prob_column].mean()),
        "brier": brier,
        "current_brier": current_brier,
        "brier_delta_vs_current": float(current_brier - brier),
        "accuracy_at_05": accuracy,
        **metrics,
    }


def _summarize(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame()
    keys = ["dataset", "split_mode", "metric_split", "model", "model_family", "conflict_bin"]
    metrics = [
        "n",
        "positive_rate",
        "mean_conflict",
        "brier",
        "brier_delta_vs_current",
        "accuracy_at_05",
        "auroc",
        "auprc",
        "ece",
        "risk_at_80_coverage",
    ]
    rows: list[dict] = []
    for key_values, group in per_run.groupby(keys, dropna=False):
        row = dict(zip(keys, key_values))
        row["runs"] = int(group["run_dir"].nunique())
        row["total_n"] = int(group["n"].sum())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = float(values.mean()) if not values.empty else float("nan")
            row[f"std_{metric}"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            row[f"ci95_{metric}"] = (
                float(1.96 * values.std(ddof=0) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def build_conflict_report(
    *,
    outputs_root: Path,
    output_dir: Path,
    patterns: list[str],
    current_model: str,
    retrieval_model: str,
    fusion_models: list[str],
    metric_splits: list[str],
    n_bins: int,
    run_dirs: list[str] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = resolve_run_dirs(outputs_root=outputs_root, patterns=patterns, explicit_run_dirs=run_dirs or [])
    per_run_rows: list[dict] = []
    sample_rows: list[dict] = []
    manifest_rows: list[dict] = []

    for run_dir in resolved:
        metrics_payload = _read_metrics(run_dir)
        run_payload_base = {
            "run_dir": run_dir.name,
            "dataset": metrics_payload.get("dataset"),
            "split_mode": metrics_payload.get("split_mode"),
            "seed": metrics_payload.get("seed", infer_seed_from_run_dir(run_dir.name)),
        }
        manifest_row = {**run_payload_base, "status": "no_usable_split", "metric_splits": ""}
        usable_splits: list[str] = []

        for metric_split in metric_splits:
            prediction_path = prediction_path_for_split(run_dir, metric_split)
            if not prediction_path.exists():
                continue
            frame = pd.read_csv(prediction_path)
            current_col = probability_column(current_model, metric_split)
            retrieval_col = probability_column(retrieval_model, metric_split)
            if current_col not in frame.columns or retrieval_col not in frame.columns or "label" not in frame.columns:
                continue

            selected_models = [current_model, retrieval_model]
            selected_models.extend(model for model in fusion_models if model not in selected_models)
            model_columns = {
                model: probability_column(model, metric_split)
                for model in selected_models
                if probability_column(model, metric_split) in frame.columns
            }
            if len(model_columns) < 2:
                continue

            keep_columns = ["label", current_col, retrieval_col, *model_columns.values()]
            metadata_columns = [
                column
                for column in ["sample_id", "drug_id", "target_id", "assay_id", "document_year", "text_missing", "sequence_missing"]
                if column in frame.columns
            ]
            working = frame[metadata_columns + sorted(set(keep_columns))].copy()
            for column in sorted(set(keep_columns)):
                working[column] = pd.to_numeric(working[column], errors="coerce")
            working = working.dropna(subset=["label", current_col, retrieval_col])
            if working.empty:
                continue
            working["label"] = working["label"].astype(int)
            working["conflict"] = (working[current_col] - working[retrieval_col]).abs()
            working["conflict_bin"] = _conflict_bins(working["conflict"], n_bins=n_bins)
            usable_splits.append(metric_split)

            sample_export = working[metadata_columns + ["label", current_col, retrieval_col, "conflict", "conflict_bin"]].copy()
            sample_export.insert(0, "metric_split", metric_split)
            for key, value in reversed(run_payload_base.items()):
                sample_export.insert(0, key, value)
            sample_export = sample_export.rename(
                columns={
                    current_col: "current_prob",
                    retrieval_col: "retrieval_prob",
                }
            )
            sample_rows.extend(sample_export.to_dict(orient="records"))

            for conflict_bin, bin_frame in working.groupby("conflict_bin", dropna=False):
                for model, prob_column in model_columns.items():
                    scored = bin_frame.dropna(subset=[prob_column, current_col])
                    if scored.empty:
                        continue
                    per_run_rows.append(
                        _metric_row(
                            frame=scored,
                            prob_column=prob_column,
                            current_column=current_col,
                            run_payload={**run_payload_base, "metric_split": metric_split},
                            model=model,
                            model_family=_safe_model_family(model, current_model, retrieval_model),
                            conflict_bin=int(conflict_bin),
                        )
                    )

        if usable_splits:
            manifest_row["status"] = "analyzed"
            manifest_row["metric_splits"] = ",".join(sorted(set(usable_splits)))
        manifest_rows.append(manifest_row)

    if not per_run_rows:
        raise ValueError("No conflict rows were generated from the provided runs.")

    per_run = pd.DataFrame(per_run_rows)
    summary = _summarize(per_run)
    samples = pd.DataFrame(sample_rows)
    manifest = pd.DataFrame(manifest_rows)

    per_run.to_csv(output_dir / "conflict_per_run.csv", index=False)
    summary.to_csv(output_dir / "conflict_summary.csv", index=False)
    samples.to_csv(output_dir / "conflict_sample_scores.csv", index=False)
    manifest.to_csv(output_dir / "run_manifest.csv", index=False)

    report = {
        "runs_considered": int(len(resolved)),
        "runs_analyzed": int(per_run["run_dir"].nunique()),
        "rows_written": int(len(per_run)),
        "sample_rows_written": int(len(samples)),
        "current_model": current_model,
        "retrieval_model": retrieval_model,
        "fusion_models": fusion_models,
        "metric_splits": sorted(per_run["metric_split"].unique().tolist()),
        "models_analyzed": sorted(per_run["model"].unique().tolist()),
        "output_dir": str(output_dir),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = build_conflict_report(
        outputs_root=(REPO_ROOT / args.outputs_root).resolve(),
        output_dir=(REPO_ROOT / args.output_dir).resolve(),
        patterns=args.patterns,
        current_model=args.current_model,
        retrieval_model=args.retrieval_model,
        fusion_models=args.fusion_models,
        metric_splits=args.metric_splits,
        n_bins=args.n_bins,
        run_dirs=args.run_dirs,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
