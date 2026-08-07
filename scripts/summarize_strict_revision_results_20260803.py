#!/usr/bin/env python3
"""Summarize strict revision experiments for the MIRAGE-DTA revision."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)


ROOT = Path("/home/test/wsk/mirage_drug_miniprototype")
OUT = ROOT / "outputs" / "revision_strict_summary_20260803"
OUT.mkdir(parents=True, exist_ok=True)

COMPONENT_35_DIRS = [
    ROOT / "outputs" / "revision_e03_chembl_full_20260729",
    ROOT / "outputs" / "revision_e16_component_10seed_extension_20260803",
]
COMPONENT_0_DIRS = [ROOT / "outputs" / "revision_e19_component_nomissing_20260803"]
EXTERNAL_DIRS = [
    ROOT / "outputs" / "revision_e17_external_chembl_strict_20260803",
    ROOT / "outputs" / "revision_e18_external_chembl_strict_mgraphdta_20260803",
]

MODEL_COLUMNS = {
    "MIRAGE-DTA": "mirage_full",
    "MIRAGE-DTA w/o gate": "mirage_w_o_gate",
    "MIRAGE-DTA w/o probe": "mirage_w_o_probe",
    "MIRAGE-DTA w/o anchor": "mirage_w_o_anchor",
    "Blend average": "hybrid_blend_avg",
    "No-mask concat": "no_mask",
    "Mask-aware current": "mask",
    "Retrieval-only": "historical_retrieval_evidence",
}


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            idx = (p >= lo) & (p <= hi)
        else:
            idx = (p >= lo) & (p < hi)
        if not np.any(idx):
            continue
        conf = np.mean(p[idx])
        acc = np.mean(y[idx])
        out += np.mean(idx) * abs(acc - conf)
    return float(out)


def selective_risk_at_80(y: np.ndarray, p: np.ndarray) -> float:
    confidence = np.abs(p - 0.5)
    n_keep = max(1, int(math.ceil(0.8 * len(y))))
    keep = np.argsort(-confidence)[:n_keep]
    return float(np.mean((p[keep] >= 0.5) != y[keep]))


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = y.astype(int)
    p = np.clip(p.astype(float), 1e-7, 1 - 1e-7)
    prevalence = float(np.mean(y))
    pred = (p >= 0.5).astype(int)
    out = {
        "n_test": float(len(y)),
        "prevalence": prevalence,
        "auprc": float(average_precision_score(y, p)),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
        "ece": ece_score(y, p),
        "risk_at_80": selective_risk_at_80(y, p),
        "brier": float(brier_score_loss(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }
    out["pr_lift"] = out["auprc"] - prevalence
    return out


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def ci95(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    se = float(np.std(arr, ddof=1) / math.sqrt(len(arr)))
    mean = float(np.mean(arr))
    return mean - 1.96 * se, mean + 1.96 * se


def iter_prediction_files(base_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for base in base_dirs:
        if not base.exists():
            continue
        files.extend(sorted(base.glob("*_s*/predictions.csv")))
        files.extend(sorted(base.glob("*_s*_root/*_s*/predictions.csv")))
    return sorted(set(files))


def summarize_components(base_dirs: list[Path], suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run = []
    for pred_path in iter_prediction_files(base_dirs):
        df = pd.read_csv(pred_path)
        y = df["label"].to_numpy()
        parts = pred_path.parent.name.rsplit("_s", 1)
        if len(parts) != 2:
            continue
        split, seed_text = parts
        try:
            seed = int(seed_text)
        except ValueError:
            continue
        for model_name, stem in MODEL_COLUMNS.items():
            col = f"{stem}_prob_{suffix}"
            if col not in df.columns:
                continue
            row = {
                "split": split,
                "seed": seed,
                "model": model_name,
                "probability_column": col,
            }
            row.update(classification_metrics(y, df[col].to_numpy()))
            per_run.append(row)
    per_run_df = pd.DataFrame(per_run)
    if per_run_df.empty:
        return per_run_df, per_run_df
    per_run_df = per_run_df.drop_duplicates(["split", "seed", "model"], keep="last")
    per_run_df.to_csv(OUT / f"component_{suffix}_per_run.csv", index=False)

    rows = []
    metric_cols = [
        "n_test",
        "prevalence",
        "auprc",
        "pr_lift",
        "auroc",
        "ece",
        "risk_at_80",
        "brier",
        "balanced_accuracy",
        "mcc",
    ]
    for (split, model), group in per_run_df.groupby(["split", "model"], sort=True):
        out = {"split": split, "model": model, "n_runs": len(group)}
        for metric in metric_cols:
            mean, std = mean_std(group[metric].tolist())
            lo, hi = ci95(group[metric].tolist())
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_ci_low"] = lo
            out[f"{metric}_ci_high"] = hi
        rows.append(out)
    by_split = pd.DataFrame(rows)
    by_split.to_csv(OUT / f"component_{suffix}_by_split_model.csv", index=False)

    rows = []
    for model, group in per_run_df.groupby("model", sort=True):
        split_means = []
        for split, split_group in group.groupby("split"):
            rec = {"split": split}
            for metric in metric_cols:
                rec[metric] = float(np.mean(split_group[metric]))
            split_means.append(rec)
        split_df = pd.DataFrame(split_means)
        out = {"model": model, "n_split_means": len(split_df)}
        for metric in metric_cols:
            mean, std = mean_std(split_df[metric].tolist())
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        rows.append(out)
    overall = pd.DataFrame(rows)
    overall.to_csv(OUT / f"component_{suffix}_overall.csv", index=False)
    write_component_deltas(per_run_df, suffix)
    return per_run_df, by_split


def write_component_deltas(per_run_df: pd.DataFrame, suffix: str) -> None:
    if per_run_df.empty:
        return
    metrics = ["auprc", "auroc", "ece", "risk_at_80", "brier", "mcc"]
    baselines = ["MIRAGE-DTA", "Blend average"]
    rows = []
    key_cols = ["split", "seed"]
    for baseline in baselines:
        base = per_run_df[per_run_df["model"] == baseline][key_cols + metrics].copy()
        if base.empty:
            continue
        base = base.rename(columns={m: f"{m}_baseline" for m in metrics})
        for model, group in per_run_df.groupby("model", sort=True):
            if model == baseline:
                continue
            merged = group.merge(base, on=key_cols, how="inner")
            if merged.empty:
                continue
            for split, split_group in merged.groupby("split", sort=True):
                out = {
                    "baseline": baseline,
                    "model": model,
                    "split": split,
                    "n_pairs": len(split_group),
                }
                for metric in metrics:
                    delta = split_group[metric] - split_group[f"{metric}_baseline"]
                    mean, std = mean_std(delta.tolist())
                    lo, hi = ci95(delta.tolist())
                    out[f"delta_{metric}_mean"] = mean
                    out[f"delta_{metric}_std"] = std
                    out[f"delta_{metric}_ci_low"] = lo
                    out[f"delta_{metric}_ci_high"] = hi
                rows.append(out)
    pd.DataFrame(rows).to_csv(OUT / f"component_{suffix}_paired_deltas.csv", index=False)


def summarize_external() -> pd.DataFrame:
    rows = []
    for base in EXTERNAL_DIRS:
        if not base.exists():
            continue
        for path in sorted(base.glob("*/external_metrics.json")):
            d = json.load(open(path))
            metrics = d.get("metrics", {})
            meta = d.get("run_meta", {})
            rows.append(
                {
                    "framework": d.get("framework"),
                    "model": d.get("model"),
                    "split": d.get("split_mode"),
                    "seed": d.get("seed"),
                    "auprc": metrics.get("auprc"),
                    "auroc": metrics.get("auroc"),
                    "rmse": metrics.get("rmse"),
                    "mae": metrics.get("mae"),
                    "pearson": metrics.get("pearson"),
                    "spearman": metrics.get("spearman"),
                    "ci": metrics.get("ci"),
                    "params": meta.get("parameter_count"),
                    "train_seconds": meta.get("train_seconds"),
                    "inference_seconds_per_1000": meta.get("inference_seconds_per_1000"),
                    "peak_gpu_memory_mb": meta.get("peak_gpu_memory_mb"),
                }
            )
    per_run = pd.DataFrame(rows)
    if per_run.empty:
        return per_run
    per_run.to_csv(OUT / "external_strict_chembl_per_run.csv", index=False)
    metric_cols = [
        "auprc",
        "auroc",
        "rmse",
        "mae",
        "pearson",
        "spearman",
        "ci",
        "params",
        "train_seconds",
        "inference_seconds_per_1000",
        "peak_gpu_memory_mb",
    ]
    agg_rows = []
    for (model, split), group in per_run.groupby(["model", "split"], sort=True):
        out = {"model": model, "split": split, "n_runs": len(group)}
        for metric in metric_cols:
            mean, std = mean_std(group[metric].tolist())
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        agg_rows.append(out)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT / "external_strict_chembl_by_split_model.csv", index=False)
    return agg


def write_component_latex(overall: pd.DataFrame, suffix: str) -> None:
    if overall.empty:
        return
    order = [
        "MIRAGE-DTA",
        "MIRAGE-DTA w/o gate",
        "MIRAGE-DTA w/o probe",
        "MIRAGE-DTA w/o anchor",
        "Blend average",
        "No-mask concat",
        "Mask-aware current",
        "Retrieval-only",
    ]
    rows = []
    for model in order:
        recs = overall[overall["model"] == model]
        if recs.empty:
            continue
        r = recs.iloc[0]
        rows.append(
            f"{model} & {r['auprc_mean']:.4f} & {r['pr_lift_mean']:.4f} & "
            f"{r['auroc_mean']:.4f} & {r['ece_mean']:.4f} & {r['risk_at_80_mean']:.4f} & "
            f"{r['brier_mean']:.4f} & {r['mcc_mean']:.4f} \\\\"
        )
    (OUT / f"component_{suffix}_table_rows.tex").write_text("\n".join(rows) + "\n")


def main() -> None:
    _, by_split_35 = summarize_components(COMPONENT_35_DIRS, "missing")
    overall_35 = pd.read_csv(OUT / "component_missing_overall.csv") if (OUT / "component_missing_overall.csv").exists() else pd.DataFrame()
    write_component_latex(overall_35, "missing")

    _, by_split_0 = summarize_components(COMPONENT_0_DIRS, "clean")
    overall_0 = pd.read_csv(OUT / "component_clean_overall.csv") if (OUT / "component_clean_overall.csv").exists() else pd.DataFrame()
    write_component_latex(overall_0, "clean")

    external = summarize_external()
    print("Wrote", OUT)
    if not overall_35.empty:
        print("component_missing_overall")
        print(overall_35[["model", "n_split_means", "auprc_mean", "auroc_mean", "ece_mean", "risk_at_80_mean", "brier_mean", "mcc_mean"]].to_string(index=False))
    if not by_split_35.empty:
        print("component_missing_by_split rows", len(by_split_35))
    if not external.empty:
        print("external_strict rows", len(external))
        print(external[["model", "split", "n_runs", "auprc_mean", "auroc_mean", "rmse_mean", "ci_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
