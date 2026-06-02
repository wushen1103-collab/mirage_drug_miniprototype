from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import shorten

import numpy as np
import pandas as pd


DEFAULT_SPLITS = ["assay_cold", "target_cold", "temporal"]
DEFAULT_FRAMEWORKS = ["deepdta", "mgraphdta"]
DEFAULT_SEEDS = [42, 43, 44]
PKD_ACTIVE_THRESHOLD = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--benchmark-frame", default="outputs/reports/chembl_descriptives/benchmark_frame.csv")
    parser.add_argument("--output-dir", default="outputs/reports/chembl_case_studies")
    parser.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS)
    parser.add_argument("--frameworks", nargs="*", default=DEFAULT_FRAMEWORKS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _run_dirs(outputs_root: Path) -> list[Path]:
    return sorted(path for path in outputs_root.glob("external_chembl_*") if path.is_dir())


def load_external_metrics(outputs_root: Path, splits: list[str], frameworks: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    split_set = set(splits)
    framework_set = set(frameworks)
    for run_dir in _run_dirs(outputs_root):
        metrics_path = run_dir / "external_metrics.json"
        if not metrics_path.exists():
            continue
        payload = _load_json(metrics_path)
        split_mode = payload.get("split_mode")
        framework = payload.get("framework")
        if split_mode not in split_set or framework not in framework_set:
            continue
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "run_dir": run_dir.name,
                "framework": framework,
                "model": payload.get("model", framework),
                "dataset": payload.get("dataset"),
                "split_mode": split_mode,
                "seed": payload.get("seed"),
                "metric_split": payload.get("metric_split", "test"),
                "has_prediction_file": bool((run_dir / "test_prediction_with_binary.csv").exists()),
                **{key: metrics.get(key) for key in ["auroc", "auprc", "rmse", "pearson", "spearman", "ci"]},
            }
        )
    return pd.DataFrame(rows)


def summarize_external_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    metric_columns = ["auroc", "auprc", "rmse", "pearson", "spearman", "ci"]
    for key_values, group in metrics.groupby(["split_mode", "framework", "model"], dropna=False):
        row = dict(zip(["split_mode", "framework", "model"], key_values))
        row["runs"] = int(len(group))
        row["seeds"] = ",".join(str(int(seed)) for seed in sorted(pd.to_numeric(group["seed"], errors="coerce").dropna()))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"mean_{metric}"] = float(values.mean()) if not values.empty else float("nan")
            row[f"std_{metric}"] = float(values.std(ddof=0)) if len(values) > 1 else 0.0
            row[f"ci95_{metric}"] = (
                float(1.96 * values.std(ddof=0) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split_mode", "framework", "model"]).reset_index(drop=True)


def seed_gap_frame(
    *,
    outputs_root: Path,
    metrics: pd.DataFrame,
    splits: list[str],
    frameworks: list[str],
    seeds: list[int],
) -> pd.DataFrame:
    present_metrics = set()
    present_predictions = set()
    if not metrics.empty:
        for row in metrics.to_dict(orient="records"):
            key = (str(row["split_mode"]), str(row["framework"]), int(row["seed"]))
            present_metrics.add(key)
            if bool(row.get("has_prediction_file")):
                present_predictions.add(key)

    rows = []
    for split_mode in splits:
        for framework in frameworks:
            for seed in seeds:
                key = (split_mode, framework, seed)
                run_dir = outputs_root / f"external_chembl_{framework}_{split_mode}_s{seed}"
                rows.append(
                    {
                        "split_mode": split_mode,
                        "framework": framework,
                        "seed": int(seed),
                        "expected_run_dir": run_dir.name,
                        "has_metrics": key in present_metrics,
                        "has_predictions": key in present_predictions,
                        "is_present": key in present_metrics and key in present_predictions,
                    }
                )
    return pd.DataFrame(rows)


def _load_benchmark_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["sample_id"])
    frame = pd.read_csv(path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"Benchmark frame has no sample_id column: {path}")
    return frame


def load_external_predictions(outputs_root: Path, metrics: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if metrics.empty:
        return pd.DataFrame()
    metadata_columns = [
        column
        for column in [
            "sample_id",
            "assay_id",
            "document_year",
            "smiles",
            "assay_text",
            "text",
            "target_text",
            "label",
            "label_raw",
        ]
        if column in benchmark.columns
    ]
    metadata = benchmark[metadata_columns].copy() if metadata_columns else pd.DataFrame(columns=["sample_id"])

    for record in metrics.to_dict(orient="records"):
        prediction_path = outputs_root / str(record["run_dir"]) / "test_prediction_with_binary.csv"
        if not prediction_path.exists():
            continue
        predictions = pd.read_csv(prediction_path)
        if "sample_id" not in predictions.columns or "pkd_prediction" not in predictions.columns:
            continue
        if "pkd_label" not in predictions.columns:
            if "Y" in predictions.columns:
                predictions["pkd_label"] = predictions["Y"]
            else:
                continue
        if "binary_label" not in predictions.columns:
            predictions["binary_label"] = (pd.to_numeric(predictions["pkd_label"], errors="coerce") >= PKD_ACTIVE_THRESHOLD).astype(int)
        predictions["framework"] = record["framework"]
        predictions["model"] = record["model"]
        predictions["split_mode"] = record["split_mode"]
        predictions["seed"] = record["seed"]
        predictions["run_dir"] = record["run_dir"]
        predictions["pkd_label"] = pd.to_numeric(predictions["pkd_label"], errors="coerce")
        predictions["pkd_prediction"] = pd.to_numeric(predictions["pkd_prediction"], errors="coerce")
        predictions["prediction_error"] = (predictions["pkd_prediction"] - predictions["pkd_label"]).abs()
        predictions["signed_error"] = predictions["pkd_prediction"] - predictions["pkd_label"]
        predictions["predicted_binary"] = (predictions["pkd_prediction"] >= PKD_ACTIVE_THRESHOLD).astype(int)
        predictions["binary_label"] = pd.to_numeric(predictions["binary_label"], errors="coerce").astype("Int64")
        predictions["binary_correct"] = predictions["predicted_binary"].eq(predictions["binary_label"])
        if not metadata.empty:
            predictions = predictions.merge(metadata, on="sample_id", how="left", suffixes=("", "_benchmark"))
        rows.append(predictions)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_case_candidates(predictions: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    base_columns = [
        "sample_id",
        "drug_id",
        "target_id",
        "assay_id",
        "document_year",
        "framework",
        "model",
        "split_mode",
        "seed",
        "binary_label",
        "predicted_binary",
        "binary_correct",
        "label_raw_nm",
        "pkd_label",
        "pkd_prediction",
        "prediction_error",
        "signed_error",
        "assay_text",
        "target_text",
        "smiles",
    ]
    available_base_columns = [column for column in base_columns if column in predictions.columns]

    for _, group in predictions.groupby(["split_mode", "framework"], dropna=False):
        selected = group.sort_values("prediction_error", ascending=False).head(top_n)
        for record in selected[available_base_columns].to_dict(orient="records"):
            rows.append({"case_type": "large_regression_error", **record})

        if "binary_correct" in group.columns:
            failed = group[group["binary_correct"].eq(False)].sort_values("prediction_error", ascending=False).head(top_n)
            for record in failed[available_base_columns].to_dict(orient="records"):
                rows.append({"case_type": "binary_decision_failure", **record})

    if {"split_mode", "seed", "sample_id", "predicted_binary"}.issubset(predictions.columns):
        for key_values, group in predictions.groupby(["split_mode", "seed", "sample_id"], dropna=False):
            if group["predicted_binary"].nunique(dropna=True) < 2 and group["pkd_prediction"].max() - group["pkd_prediction"].min() < 1.0:
                continue
            representative = group.sort_values("prediction_error", ascending=False).iloc[0]
            record = {column: representative.get(column) for column in available_base_columns}
            record.update(
                {
                    "case_type": "cross_model_disagreement",
                    "split_mode": key_values[0],
                    "seed": key_values[1],
                    "sample_id": key_values[2],
                    "framework": ",".join(sorted(group["framework"].dropna().astype(str).unique())),
                    "model": ",".join(sorted(group["model"].dropna().astype(str).unique())),
                    "pkd_prediction_min": float(group["pkd_prediction"].min()),
                    "pkd_prediction_max": float(group["pkd_prediction"].max()),
                    "prediction_error_max": float(group["prediction_error"].max()),
                }
            )
            rows.append(record)

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    sort_columns = [column for column in ["case_type", "split_mode", "prediction_error", "prediction_error_max"] if column in frame.columns]
    return frame.sort_values(sort_columns, ascending=[True, True, False, False][: len(sort_columns)]).reset_index(drop=True)


def _short_text(value: object, width: int = 180) -> str:
    if pd.isna(value):
        return ""
    return shorten(str(value).replace("\n", " "), width=width, placeholder="...")


def write_notes(
    *,
    path: Path,
    metrics_summary: pd.DataFrame,
    seed_gaps: pd.DataFrame,
    cases: pd.DataFrame,
    top_n: int,
) -> None:
    lines = [
        "# CHEMBL_ASSAY Case Study Notes",
        "",
        "This file is generated without plotting. It collects external-baseline seed coverage and sample-level cases that can be converted into manuscript examples later.",
        "",
        "## External Baseline Coverage",
    ]
    if seed_gaps.empty:
        lines.append("No seed-gap table was available.")
    else:
        missing = seed_gaps[seed_gaps["is_present"].eq(False)]
        lines.append(f"Expected combinations: {len(seed_gaps)}; complete combinations: {int(seed_gaps['is_present'].sum())}; missing combinations: {len(missing)}.")
        if not missing.empty:
            for record in missing.head(16).to_dict(orient="records"):
                lines.append(f"- Missing {record['framework']} {record['split_mode']} seed {record['seed']} ({record['expected_run_dir']}).")

    lines.extend(["", "## Metric Snapshot"])
    if metrics_summary.empty:
        lines.append("No external metrics were found.")
    else:
        display_columns = [
            column
            for column in ["split_mode", "framework", "runs", "seeds", "mean_auprc", "mean_auroc", "mean_rmse", "mean_ci"]
            if column in metrics_summary.columns
        ]
        lines.append(metrics_summary[display_columns].to_markdown(index=False))

    lines.extend(["", "## Candidate Cases"])
    if cases.empty:
        lines.append("No case candidates were generated.")
    else:
        for case_type, group in cases.groupby("case_type", dropna=False):
            lines.append(f"### {case_type}")
            for record in group.head(top_n).to_dict(orient="records"):
                sample_id = record.get("sample_id", "")
                split_mode = record.get("split_mode", "")
                framework = record.get("framework", "")
                seed = record.get("seed", "")
                label = record.get("binary_label", "")
                pred = record.get("predicted_binary", "")
                pkd_label = record.get("pkd_label", float("nan"))
                pkd_pred = record.get("pkd_prediction", float("nan"))
                error = record.get("prediction_error", record.get("prediction_error_max", float("nan")))
                lines.append(
                    f"- `{sample_id}` | {split_mode} | {framework} seed {seed} | label={label}, pred={pred}, "
                    f"pKd true={pkd_label:.3f}, pKd pred={pkd_pred:.3f}, abs error={error:.3f}"
                )
                assay_text = _short_text(record.get("assay_text", ""))
                target_text = _short_text(record.get("target_text", ""))
                if assay_text:
                    lines.append(f"  Assay: {assay_text}")
                if target_text:
                    lines.append(f"  Target: {target_text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_chembl_case_studies(
    *,
    outputs_root: Path,
    benchmark_frame: Path,
    output_dir: Path,
    splits: list[str],
    frameworks: list[str],
    seeds: list[int],
    top_n: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = _load_benchmark_frame(benchmark_frame)
    metrics = load_external_metrics(outputs_root=outputs_root, splits=splits, frameworks=frameworks)
    metrics_summary = summarize_external_metrics(metrics)
    seed_gaps = seed_gap_frame(
        outputs_root=outputs_root,
        metrics=metrics,
        splits=splits,
        frameworks=frameworks,
        seeds=seeds,
    )
    predictions = load_external_predictions(outputs_root=outputs_root, metrics=metrics, benchmark=benchmark)
    cases = build_case_candidates(predictions=predictions, top_n=top_n)

    metrics.to_csv(output_dir / "chembl_external_metrics_long.csv", index=False)
    metrics_summary.to_csv(output_dir / "chembl_external_metrics_summary.csv", index=False)
    seed_gaps.to_csv(output_dir / "chembl_external_seed_gaps.csv", index=False)
    predictions.to_csv(output_dir / "chembl_external_predictions_long.csv", index=False)
    cases.to_csv(output_dir / "chembl_case_study_candidates.csv", index=False)
    write_notes(
        path=output_dir / "chembl_case_study_notes.md",
        metrics_summary=metrics_summary,
        seed_gaps=seed_gaps,
        cases=cases,
        top_n=top_n,
    )

    report = {
        "metric_rows": int(len(metrics)),
        "prediction_rows": int(len(predictions)),
        "case_rows": int(len(cases)),
        "expected_seed_combinations": int(len(seed_gaps)),
        "complete_seed_combinations": int(seed_gaps["is_present"].sum()) if not seed_gaps.empty else 0,
        "missing_seed_combinations": int(seed_gaps["is_present"].eq(False).sum()) if not seed_gaps.empty else 0,
        "splits": splits,
        "frameworks": frameworks,
        "seeds": seeds,
        "output_dir": str(output_dir),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    report = build_chembl_case_studies(
        outputs_root=(repo_root / args.outputs_root).resolve(),
        benchmark_frame=(repo_root / args.benchmark_frame).resolve(),
        output_dir=(repo_root / args.output_dir).resolve(),
        splits=args.splits,
        frameworks=args.frameworks,
        seeds=args.seeds,
        top_n=args.top_n,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
