from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


SUMMARY_COLUMNS = [
    "split_mode",
    "missing_sequence_prob",
    "missing_text_prob",
    "model",
    "mean_test_missing_auroc",
    "mean_test_missing_auprc",
    "mean_test_missing_ece",
    "mean_test_missing_risk80",
]

OFFICIAL_RUN_COLUMNS = [
    "run_dir",
    "seed",
    "dataset",
    "split_mode",
    "split_source",
    "metric_split",
    "model",
    "auprc",
    "auroc",
    "ece",
    "risk_at_80_coverage",
    "blend_alpha",
    "selected_model",
]

EXTERNAL_RUN_COLUMNS = [
    "run_dir",
    "framework",
    "model",
    "dataset",
    "split_mode",
    "metric_split",
    "seed",
    "auprc",
    "auroc",
    "rmse",
    "pearson",
    "spearman",
    "ci",
]

SEED_PATTERN = re.compile(r"(?:^|_)s(?P<seed>\d+)(?:_|$)")


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


def load_summary_tables(paths: Iterable[str | Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = set(SUMMARY_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Summary table {path} is missing columns: {sorted(missing)}")
        frames.append(frame[SUMMARY_COLUMNS].copy())
    if not frames:
        raise ValueError("At least one summary table is required")
    combined = pd.concat(frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    return combined


def infer_seed_from_run_dir(run_dir: str) -> int | None:
    match = SEED_PATTERN.search(run_dir)
    if not match:
        return None
    return int(match.group("seed"))


def _load_official_metrics_rows(metrics_path: Path, metric_split: str) -> list[dict]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows = []
    run_dir = metrics_path.parent.name
    inferred_seed = payload.get("seed", infer_seed_from_run_dir(run_dir))
    for model_name, model_metrics in payload["models"].items():
        split_metrics = model_metrics.get(metric_split)
        if not isinstance(split_metrics, dict):
            continue
        rows.append(
            {
                "run_dir": run_dir,
                "seed": inferred_seed,
                "dataset": payload.get("dataset"),
                "split_mode": payload.get("split_mode"),
                "split_source": payload.get("split_source"),
                "metric_split": metric_split,
                "model": model_name,
                "auprc": split_metrics.get("auprc"),
                "auroc": split_metrics.get("auroc"),
                "ece": split_metrics.get("ece"),
                "risk_at_80_coverage": split_metrics.get("risk_at_80_coverage"),
                "blend_alpha": model_metrics.get("blend_alpha"),
                "selected_model": model_metrics.get("selected_model"),
            }
        )
    return rows


def load_official_run_tables(
    outputs_root: str | Path,
    patterns: Sequence[str],
    metric_split: str = "test_clean",
) -> pd.DataFrame:
    outputs_root = Path(outputs_root)
    run_dirs: list[Path] = []
    for pattern in patterns:
        run_dirs.extend(sorted(outputs_root.glob(pattern)))
    unique_run_dirs = sorted({run_dir.resolve(): run_dir for run_dir in run_dirs}.values(), key=lambda p: p.name)
    if not unique_run_dirs:
        raise ValueError(f"No run directories matched patterns: {list(patterns)}")

    rows: list[dict] = []
    for run_dir in unique_run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        rows.extend(_load_official_metrics_rows(metrics_path, metric_split=metric_split))
    if not rows:
        raise ValueError("Matched run directories, but found no readable metrics rows.")

    frame = pd.DataFrame(rows)
    return frame[OFFICIAL_RUN_COLUMNS].sort_values(["split_mode", "run_dir", "model"]).reset_index(drop=True)


def summarize_official_run_tables(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"dataset", "split_mode", "metric_split", "model", "auprc", "auroc", "ece", "risk_at_80_coverage"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Official run table is missing columns: {sorted(missing)}")
    summary = (
        frame.groupby(["dataset", "split_mode", "metric_split", "model"], dropna=False)
        .agg(
            runs=("run_dir", "count"),
            mean_auprc=("auprc", lambda x: _summary_stat(x, "mean")),
            std_auprc=("auprc", lambda x: _summary_stat(x, "std")),
            ci95_auprc=("auprc", lambda x: _summary_stat(x, "ci95")),
            mean_auroc=("auroc", lambda x: _summary_stat(x, "mean")),
            std_auroc=("auroc", lambda x: _summary_stat(x, "std")),
            ci95_auroc=("auroc", lambda x: _summary_stat(x, "ci95")),
            mean_ece=("ece", lambda x: _summary_stat(x, "mean")),
            std_ece=("ece", lambda x: _summary_stat(x, "std")),
            ci95_ece=("ece", lambda x: _summary_stat(x, "ci95")),
            mean_risk80=("risk_at_80_coverage", lambda x: _summary_stat(x, "mean")),
            std_risk80=("risk_at_80_coverage", lambda x: _summary_stat(x, "std")),
            ci95_risk80=("risk_at_80_coverage", lambda x: _summary_stat(x, "ci95")),
        )
        .reset_index()
        .sort_values(["split_mode", "mean_auprc", "mean_auroc"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return summary


def compute_selection_instability(frame: pd.DataFrame, selector_model: str = "fullsuite_val_select") -> pd.DataFrame:
    required = {"run_dir", "seed", "dataset", "split_mode", "metric_split", "model", "auprc", "auroc", "selected_model"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Official run table is missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for (run_dir, split_mode, metric_split), run_frame in frame.groupby(["run_dir", "split_mode", "metric_split"], dropna=False):
        selector_rows = run_frame[run_frame["model"] == selector_model]
        if selector_rows.empty:
            continue
        selector_row = selector_rows.iloc[0]
        selected_name = selector_row["selected_model"]
        if not selected_name:
            continue

        candidates = run_frame[run_frame["model"] != selector_model].copy()
        if candidates.empty:
            continue
        best_row = candidates.sort_values(["auprc", "auroc"], ascending=[False, False]).iloc[0]
        selected_rows = candidates[candidates["model"] == selected_name]
        if selected_rows.empty:
            continue
        selected_row = selected_rows.iloc[0]
        rows.append(
            {
                "run_dir": run_dir,
                "seed": selector_row["seed"],
                "dataset": selector_row["dataset"],
                "split_mode": split_mode,
                "metric_split": metric_split,
                "selector_model": selector_model,
                "selected_model": selected_name,
                "best_test_model": best_row["model"],
                "selected_test_auprc": float(selected_row["auprc"]),
                "best_test_auprc": float(best_row["auprc"]),
                "selected_test_auroc": float(selected_row["auroc"]),
                "best_test_auroc": float(best_row["auroc"]),
                "test_auprc_gap": round(float(best_row["auprc"] - selected_row["auprc"]), 12),
                "test_auroc_gap": round(float(best_row["auroc"] - selected_row["auroc"]), 12),
                "selected_matches_test_best": bool(selected_name == best_row["model"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["split_mode", "run_dir"]).reset_index(drop=True)


def summarize_selection_instability(instability: pd.DataFrame) -> pd.DataFrame:
    if instability.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "split_mode",
                "metric_split",
                "runs",
                "match_rate",
                "mean_test_auprc_gap",
                "max_test_auprc_gap",
                "mean_test_auroc_gap",
                "max_test_auroc_gap",
            ]
        )
    summary = (
        instability.groupby(["dataset", "split_mode", "metric_split"], dropna=False)
        .agg(
            runs=("run_dir", "count"),
            match_rate=("selected_matches_test_best", "mean"),
            mean_test_auprc_gap=("test_auprc_gap", "mean"),
            max_test_auprc_gap=("test_auprc_gap", "max"),
            mean_test_auroc_gap=("test_auroc_gap", "mean"),
            max_test_auroc_gap=("test_auroc_gap", "max"),
        )
        .reset_index()
        .sort_values(["split_mode", "metric_split"])
        .reset_index(drop=True)
    )
    return summary


def _load_external_metrics_row(metrics_path: Path) -> dict:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    return {
        "run_dir": metrics_path.parent.name,
        "framework": payload.get("framework"),
        "model": payload.get("model"),
        "dataset": payload.get("dataset"),
        "split_mode": payload.get("split_mode"),
        "metric_split": payload.get("metric_split", "test_clean"),
        "seed": payload.get("seed", infer_seed_from_run_dir(metrics_path.parent.name)),
        "auprc": metrics.get("auprc"),
        "auroc": metrics.get("auroc"),
        "rmse": metrics.get("rmse"),
        "pearson": metrics.get("pearson"),
        "spearman": metrics.get("spearman"),
        "ci": metrics.get("ci"),
    }


def load_external_run_tables(outputs_root: str | Path, patterns: Sequence[str]) -> pd.DataFrame:
    outputs_root = Path(outputs_root)
    run_dirs: list[Path] = []
    for pattern in patterns:
        run_dirs.extend(sorted(outputs_root.glob(pattern)))
    unique_run_dirs = sorted({run_dir.resolve(): run_dir for run_dir in run_dirs}.values(), key=lambda p: p.name)
    if not unique_run_dirs:
        raise ValueError(f"No external run directories matched patterns: {list(patterns)}")

    rows: list[dict] = []
    for run_dir in unique_run_dirs:
        metrics_path = run_dir / "external_metrics.json"
        if not metrics_path.exists():
            continue
        rows.append(_load_external_metrics_row(metrics_path))
    if not rows:
        raise ValueError("Matched external run directories, but found no readable external_metrics.json files.")

    frame = pd.DataFrame(rows)
    return frame[EXTERNAL_RUN_COLUMNS].sort_values(
        ["framework", "model", "split_mode", "run_dir"]
    ).reset_index(drop=True)


def summarize_external_run_tables(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "framework",
        "model",
        "dataset",
        "split_mode",
        "metric_split",
        "auprc",
        "auroc",
        "rmse",
        "pearson",
        "spearman",
        "ci",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"External run table is missing columns: {sorted(missing)}")
    summary = (
        frame.groupby(["framework", "model", "dataset", "split_mode", "metric_split"], dropna=False)
        .agg(
            runs=("run_dir", "count"),
            mean_auprc=("auprc", lambda x: _summary_stat(x, "mean")),
            std_auprc=("auprc", lambda x: _summary_stat(x, "std")),
            ci95_auprc=("auprc", lambda x: _summary_stat(x, "ci95")),
            mean_auroc=("auroc", lambda x: _summary_stat(x, "mean")),
            std_auroc=("auroc", lambda x: _summary_stat(x, "std")),
            ci95_auroc=("auroc", lambda x: _summary_stat(x, "ci95")),
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
        .sort_values(["split_mode", "mean_auprc", "mean_auroc"], ascending=[True, False, False])
        .reset_index(drop=True)
    )
    return summary


def compute_model_significance(
    frame: pd.DataFrame,
    primary_model: str,
    baseline_model: str,
    metrics: Sequence[str],
    group_cols: Sequence[str] = ("dataset", "split_mode", "metric_split"),
) -> pd.DataFrame:
    required = {"seed", "model", *metrics}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Run table is missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for group_values, condition_frame in frame.groupby(list(group_cols), dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        group_payload = dict(zip(group_cols, group_values))
        reduced = (
            condition_frame[condition_frame["model"].isin([primary_model, baseline_model])]
            .groupby(["seed", "model"], dropna=False)[list(metrics)]
            .mean()
            .reset_index()
        )
        if reduced.empty:
            continue
        for metric in metrics:
            pivot = reduced.pivot(index="seed", columns="model", values=metric).dropna()
            if primary_model not in pivot.columns or baseline_model not in pivot.columns or pivot.empty:
                continue
            deltas = pivot[primary_model] - pivot[baseline_model]
            pair_count = int(len(deltas))
            if pair_count > 1:
                ttest_pvalue = float(stats.ttest_rel(pivot[primary_model], pivot[baseline_model]).pvalue)
            else:
                ttest_pvalue = float("nan")
            if pair_count > 1 and not np.allclose(deltas.to_numpy(dtype=float), 0.0):
                try:
                    wilcoxon_pvalue = float(stats.wilcoxon(deltas.to_numpy(dtype=float)).pvalue)
                except ValueError:
                    wilcoxon_pvalue = float("nan")
            else:
                wilcoxon_pvalue = float("nan")
            rows.append(
                {
                    **group_payload,
                    "metric": metric,
                    "primary_model": primary_model,
                    "baseline_model": baseline_model,
                    "pairs": pair_count,
                    "mean_primary": float(pivot[primary_model].mean()),
                    "mean_baseline": float(pivot[baseline_model].mean()),
                    "mean_delta": float(deltas.mean()),
                    "std_delta": float(deltas.to_numpy(dtype=float).std(ddof=0)),
                    "ttest_pvalue": ttest_pvalue,
                    "wilcoxon_pvalue": wilcoxon_pvalue,
                }
            )
    return pd.DataFrame(rows).sort_values([*group_cols, "metric"]).reset_index(drop=True)


def rank_models_across_conditions(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["condition"] = frame["split_mode"] + "_" + frame["missing_sequence_prob"].astype(str)
    frame["rank_auroc"] = frame.groupby("condition")["mean_test_missing_auroc"].rank(
        ascending=False, method="average"
    )
    frame["rank_auprc"] = frame.groupby("condition")["mean_test_missing_auprc"].rank(
        ascending=False, method="average"
    )
    frame["rank_ece"] = frame.groupby("condition")["mean_test_missing_ece"].rank(
        ascending=True, method="average"
    )
    frame["rank_risk"] = frame.groupby("condition")["mean_test_missing_risk80"].rank(
        ascending=True, method="average"
    )
    ranked = (
        frame.groupby("model")
        .agg(
            mean_auroc=("mean_test_missing_auroc", "mean"),
            min_auroc=("mean_test_missing_auroc", "min"),
            mean_auprc=("mean_test_missing_auprc", "mean"),
            mean_ece=("mean_test_missing_ece", "mean"),
            mean_risk80=("mean_test_missing_risk80", "mean"),
            mean_rank_auroc=("rank_auroc", "mean"),
            mean_rank_auprc=("rank_auprc", "mean"),
            mean_rank_ece=("rank_ece", "mean"),
            mean_rank_risk=("rank_risk", "mean"),
        )
        .reset_index()
    )
    ranked["mean_rank_overall"] = ranked[
        ["mean_rank_auroc", "mean_rank_auprc", "mean_rank_ece", "mean_rank_risk"]
    ].mean(axis=1)
    return ranked.sort_values(["mean_rank_overall", "mean_auroc"], ascending=[True, False]).reset_index(drop=True)


def filter_shortlist(summary: pd.DataFrame, models: Sequence[str]) -> pd.DataFrame:
    selected = summary[summary["model"].isin(models)].copy()
    if selected.empty:
        raise ValueError("No rows matched the requested shortlist")
    return selected.sort_values(
        ["split_mode", "missing_sequence_prob", "mean_test_missing_auroc"],
        ascending=[True, True, False],
    ).reset_index(drop=True)


def compare_models(summary: pd.DataFrame, primary_model: str, baseline_model: str) -> pd.DataFrame:
    primary = summary[summary["model"] == primary_model].copy()
    baseline = summary[summary["model"] == baseline_model].copy()
    key_cols = ["split_mode", "missing_sequence_prob", "missing_text_prob"]
    merged = primary.merge(
        baseline,
        on=key_cols,
        suffixes=("_primary", "_baseline"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"No overlapping conditions found between {primary_model} and {baseline_model}")
    merged["delta_auroc"] = merged["mean_test_missing_auroc_primary"] - merged["mean_test_missing_auroc_baseline"]
    merged["delta_auprc"] = merged["mean_test_missing_auprc_primary"] - merged["mean_test_missing_auprc_baseline"]
    merged["delta_ece"] = merged["mean_test_missing_ece_primary"] - merged["mean_test_missing_ece_baseline"]
    merged["delta_risk80"] = (
        merged["mean_test_missing_risk80_primary"] - merged["mean_test_missing_risk80_baseline"]
    )
    keep = key_cols + [
        "model_primary",
        "model_baseline",
        "mean_test_missing_auroc_primary",
        "mean_test_missing_auroc_baseline",
        "delta_auroc",
        "mean_test_missing_auprc_primary",
        "mean_test_missing_auprc_baseline",
        "delta_auprc",
        "mean_test_missing_ece_primary",
        "mean_test_missing_ece_baseline",
        "delta_ece",
        "mean_test_missing_risk80_primary",
        "mean_test_missing_risk80_baseline",
        "delta_risk80",
    ]
    return merged[keep].sort_values(key_cols).reset_index(drop=True)

