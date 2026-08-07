from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_FIXED_MODEL = "mirage_full"
SIMPLE_FALLBACKS = [
    "mask",
    "hybrid_blend_avg",
    "retrieval",
    "smiles_only_retrieval",
    "morgan_smiles_retrieval",
    "hybrid_smiles_retrieval",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/submission_hardening/davis_failure_diagnostics")
    parser.add_argument("--patterns", nargs="*", default=["tdc_davis_*_vpred"])
    parser.add_argument("--fixed-model", default=DEFAULT_FIXED_MODEL)
    return parser.parse_args()


def infer_seed_from_name(name: str):
    match = re.search(r"(?:seed|_s)(\d+)", name)
    return int(match.group(1)) if match else None


def load_runs(outputs_root: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(outputs_root.glob(pattern))
    return sorted(
        {
            p.resolve()
            for p in paths
            if (p / "metrics.json").exists()
            and (p / "predictions.csv").exists()
            and (p / "validation_predictions.csv").exists()
        },
        key=lambda p: p.name,
    )


def flatten_metrics(run_dir: Path) -> tuple[list[dict], dict]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
    pred = pd.read_csv(run_dir / "predictions.csv")
    train = pd.read_csv(run_dir / "train_preview.csv") if (run_dir / "train_preview.csv").exists() else pd.DataFrame()
    val = pd.read_csv(run_dir / "validation_predictions.csv")
    base = {
        "run_dir": run_dir.name,
        "dataset": metrics.get("dataset"),
        "split_mode": metrics.get("split_mode"),
        "seed": metrics.get("seed", infer_seed_from_name(run_dir.name)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(pred)),
        "test_prevalence": float(pred["label"].mean()) if "label" in pred else np.nan,
        "unique_train_drugs": int(train["drug_id"].nunique()) if "drug_id" in train else np.nan,
        "unique_test_drugs": int(pred["drug_id"].nunique()) if "drug_id" in pred else np.nan,
        "unique_train_targets": int(train["target_id"].nunique()) if "target_id" in train else np.nan,
        "unique_test_targets": int(pred["target_id"].nunique()) if "target_id" in pred else np.nan,
    }
    rows: list[dict] = []
    for model, payload in metrics.get("models", {}).items():
        if not isinstance(payload, dict):
            continue
        val_auprc = payload.get("val", {}).get("auprc")
        for split in ["test_clean", "test_missing"]:
            split_payload = payload.get(split)
            if not isinstance(split_payload, dict):
                continue
            rows.append(
                {
                    **base,
                    "model": model,
                    "metric_split": split,
                    "val_auprc": val_auprc,
                    "auprc": split_payload.get("auprc"),
                    "auroc": split_payload.get("auroc"),
                    "ece": split_payload.get("ece"),
                    "risk_at_80_coverage": split_payload.get("risk_at_80_coverage"),
                }
            )
    return rows, base


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = (repo_root / args.outputs_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    base_rows: list[dict] = []
    for run_dir in load_runs(outputs_root, args.patterns):
        rows, base = flatten_metrics(run_dir)
        all_rows.extend(rows)
        base_rows.append(base)
    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(output_dir / "davis_model_metrics_long.csv", index=False)
    pd.DataFrame(base_rows).drop_duplicates().to_csv(output_dir / "davis_split_density.csv", index=False)

    diagnostic_rows: list[dict] = []
    if not metrics_df.empty:
        for key, group in metrics_df.groupby(["split_mode", "metric_split", "seed"], dropna=False):
            split_mode, metric_split, seed = key
            fixed = group[group["model"] == args.fixed_model]
            if fixed.empty:
                continue
            fixed_row = fixed.iloc[0]
            simple = group[group["model"].isin(SIMPLE_FALLBACKS)].sort_values(["auprc", "auroc"], ascending=[False, False])
            internal = group.sort_values(["auprc", "auroc"], ascending=[False, False])
            val_selected = group.sort_values(["val_auprc", "auprc"], ascending=[False, False])
            best_simple = simple.iloc[0] if not simple.empty else None
            best_internal = internal.iloc[0] if not internal.empty else None
            val_best = val_selected.iloc[0] if not val_selected.empty else None
            diagnostic_rows.append(
                {
                    "split_mode": split_mode,
                    "metric_split": metric_split,
                    "seed": seed,
                    "fixed_model": args.fixed_model,
                    "fixed_auprc": fixed_row["auprc"],
                    "fixed_auroc": fixed_row["auroc"],
                    "fixed_ece": fixed_row["ece"],
                    "fixed_risk80": fixed_row["risk_at_80_coverage"],
                    "best_simple_model": best_simple["model"] if best_simple is not None else "",
                    "best_simple_auprc": best_simple["auprc"] if best_simple is not None else np.nan,
                    "fixed_minus_best_simple_auprc": fixed_row["auprc"] - best_simple["auprc"] if best_simple is not None else np.nan,
                    "best_internal_model": best_internal["model"] if best_internal is not None else "",
                    "best_internal_auprc": best_internal["auprc"] if best_internal is not None else np.nan,
                    "fixed_minus_best_internal_auprc": fixed_row["auprc"] - best_internal["auprc"] if best_internal is not None else np.nan,
                    "val_best_model": val_best["model"] if val_best is not None else "",
                    "val_best_test_auprc": val_best["auprc"] if val_best is not None else np.nan,
                    "n_train": fixed_row["n_train"],
                    "n_val": fixed_row["n_val"],
                    "n_test": fixed_row["n_test"],
                    "test_prevalence": fixed_row["test_prevalence"],
                    "unique_train_drugs": fixed_row["unique_train_drugs"],
                    "unique_test_drugs": fixed_row["unique_test_drugs"],
                    "unique_train_targets": fixed_row["unique_train_targets"],
                    "unique_test_targets": fixed_row["unique_test_targets"],
                }
            )

    diag_df = pd.DataFrame(diagnostic_rows)
    diag_df.to_csv(output_dir / "davis_failure_diagnostics_per_seed.csv", index=False)
    if not diag_df.empty:
        summary = (
            diag_df.groupby(["split_mode", "metric_split"], dropna=False)
            .agg(
                runs=("seed", "nunique"),
                mean_fixed_auprc=("fixed_auprc", "mean"),
                mean_fixed_auroc=("fixed_auroc", "mean"),
                mean_fixed_ece=("fixed_ece", "mean"),
                mean_fixed_risk80=("fixed_risk80", "mean"),
                mean_fixed_minus_best_simple_auprc=("fixed_minus_best_simple_auprc", "mean"),
                mean_fixed_minus_best_internal_auprc=("fixed_minus_best_internal_auprc", "mean"),
                mean_n_train=("n_train", "mean"),
                mean_n_test=("n_test", "mean"),
                mean_unique_train_targets=("unique_train_targets", "mean"),
                mean_unique_test_targets=("unique_test_targets", "mean"),
                mean_test_prevalence=("test_prevalence", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(output_dir / "davis_failure_diagnostics_summary.csv", index=False)

    report = {
        "runs_analyzed": int(metrics_df["run_dir"].nunique()) if not metrics_df.empty else 0,
        "metric_rows": int(len(metrics_df)),
        "diagnostic_rows": int(len(diag_df)),
        "fixed_model": args.fixed_model,
        "output_dir": str(output_dir),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
