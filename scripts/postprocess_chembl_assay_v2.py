from __future__ import annotations

import argparse
import json
import re
import sys
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import split_assay_cold, split_target_cold, split_temporal  # noqa: E402
from mirage_mini.metrics import expected_calibration_error, risk_at_coverage  # noqa: E402


MODEL_LABELS = {
    "hybrid_plus_pretrained_smiles_text": "MIRAGE-DTA",
    "hybrid_blend_avg": "Blend average",
    "hybrid_plus_pretrained_smiles_text_retrieval": "MIRAGE + retrieval",
    "retrieval": "Retrieval-only",
    "mask": "Mask-aware current",
    "no_mask": "Concat/no-mask",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="outputs/chembl_assay_v2_20260505")
    parser.add_argument("--eval-dir", default="outputs/chembl_assay_v2_eval_20260505")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[100.0, 300.0, 1000.0, 10000.0])
    parser.add_argument("--fixed-model", default="hybrid_plus_pretrained_smiles_text")
    return parser.parse_args()


def _split_frame(frame: pd.DataFrame, split_mode: str, seed: int) -> dict[str, pd.DataFrame]:
    if split_mode == "assay_cold":
        return split_assay_cold(frame, seed=seed)
    if split_mode == "target_cold":
        return split_target_cold(frame, seed=seed)
    if split_mode == "temporal":
        return split_temporal(frame, seed=seed)
    raise ValueError(split_mode)


def _infer_run(run_dir: Path) -> tuple[str, int]:
    match = re.match(r"(.+)_s(\d+)$", run_dir.name)
    if not match:
        raise ValueError(f"Cannot infer split/seed from {run_dir.name}")
    return match.group(1), int(match.group(2))


def _ensure_split_previews(frame: pd.DataFrame, eval_dir: Path) -> None:
    for run_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        if not (run_dir / "metrics.json").exists() or not (run_dir / "predictions.csv").exists():
            continue
        split_mode, seed = _infer_run(run_dir)
        splits = _split_frame(frame, split_mode, seed)
        for name in ["train", "val"]:
            out = run_dir / f"{name}_preview.csv"
            if not out.exists():
                splits[name].to_csv(out, index=False)


def _metric_dict(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(prob)
    y_true = y_true[mask].astype(int)
    prob = prob[mask].astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "prevalence": float(np.mean(y_true)) if len(y_true) else np.nan,
            "auprc": np.nan,
            "auprc_lift": np.nan,
            "auroc": np.nan,
            "ece": np.nan,
            "risk_at_80_coverage": np.nan,
            "brier": np.nan,
            "balanced_accuracy": np.nan,
            "mcc": np.nan,
        }
    prevalence = float(np.mean(y_true))
    pred = (prob >= 0.5).astype(int)
    auprc = float(average_precision_score(y_true, prob))
    return {
        "prevalence": prevalence,
        "auprc": auprc,
        "auprc_lift": float(auprc - prevalence),
        "auroc": float(roc_auc_score(y_true, prob)),
        "ece": float(expected_calibration_error(y_true, prob)),
        "risk_at_80_coverage": float(risk_at_coverage(y_true, prob, coverage=0.8)),
        "brier": float(brier_score_loss(y_true, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def threshold_audit(eval_dir: Path, thresholds: list[float], fixed_model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    prob_col = f"{fixed_model}_prob_missing"
    for run_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        pred_path = run_dir / "predictions.csv"
        if not pred_path.exists():
            continue
        split_mode, seed = _infer_run(run_dir)
        pred = pd.read_csv(pred_path)
        if prob_col not in pred.columns:
            continue
        prob = pd.to_numeric(pred[prob_col], errors="coerce").to_numpy(float)
        nm = pd.to_numeric(pred["label_raw"], errors="coerce").to_numpy(float)
        for threshold in thresholds:
            y = (nm <= threshold).astype(int)
            rows.append(
                {
                    "dataset": "CHEMBL_ASSAY_V2",
                    "split_mode": split_mode,
                    "seed": seed,
                    "threshold_nm": threshold,
                    "model": fixed_model,
                    **_metric_dict(y, prob),
                }
            )
    long = pd.DataFrame(rows)
    if long.empty:
        return long, long
    summary = (
        long.groupby(["dataset", "split_mode", "threshold_nm", "model"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            prevalence_mean=("prevalence", "mean"),
            auprc_mean=("auprc", "mean"),
            auprc_lift_mean=("auprc_lift", "mean"),
            auroc_mean=("auroc", "mean"),
            ece_mean=("ece", "mean"),
            risk_at_80_mean=("risk_at_80_coverage", "mean"),
            brier_mean=("brier", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            mcc_mean=("mcc", "mean"),
        )
    )
    return long, summary


def compact_main_tables(eval_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_runs = pd.read_csv(eval_dir / "all_runs.csv")
    missing = all_runs[all_runs["metric_split"].eq("test_missing")].copy()
    missing["model_label"] = missing["model"].map(MODEL_LABELS).fillna(missing["model"])
    main = missing[missing["model"].isin(MODEL_LABELS)].copy()
    summary = (
        main.groupby(["dataset", "split_mode", "model", "model_label"], as_index=False)
        .agg(
            n_runs=("seed", "nunique"),
            n_test_mean=("n_test", "mean"),
            prevalence_mean=("test_prevalence", "mean"),
            auprc_mean=("auprc", "mean"),
            auprc_std=("auprc", "std"),
            auprc_lift_mean=("auprc_lift", "mean"),
            auroc_mean=("auroc", "mean"),
            auroc_std=("auroc", "std"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            risk_at_80_mean=("risk_at_80_coverage", "mean"),
            brier_mean=("brier", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            mcc_mean=("mcc", "mean"),
        )
    )
    return main, summary


def main() -> None:
    args = parse_args()
    benchmark_dir = (REPO_ROOT / args.benchmark_dir).resolve()
    eval_dir = (REPO_ROOT / args.eval_dir).resolve()
    frame = pd.read_csv(benchmark_dir / "benchmark_frame.csv")
    _ensure_split_previews(frame, eval_dir)

    main_long, main_summary = compact_main_tables(eval_dir)
    main_long.to_csv(eval_dir / "v2_main_long.csv", index=False)
    main_summary.to_csv(eval_dir / "v2_main_summary.csv", index=False)

    th_long, th_summary = threshold_audit(eval_dir, args.thresholds, args.fixed_model)
    th_long.to_csv(eval_dir / "v2_threshold_audit_long.csv", index=False)
    th_summary.to_csv(eval_dir / "v2_threshold_audit_summary.csv", index=False)

    manifest = {
        "benchmark_dir": str(benchmark_dir),
        "eval_dir": str(eval_dir),
        "fixed_model": args.fixed_model,
        "fixed_model_label": MODEL_LABELS.get(args.fixed_model, args.fixed_model),
        "thresholds_nm": args.thresholds,
        "main_rows": int(len(main_long)),
        "main_summary_rows": int(len(main_summary)),
        "threshold_rows": int(len(th_long)),
    }
    (eval_dir / "v2_postprocess_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(main_summary.to_string(index=False))


if __name__ == "__main__":
    main()

