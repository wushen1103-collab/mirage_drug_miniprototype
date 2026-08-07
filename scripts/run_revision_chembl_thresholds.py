from __future__ import annotations

import argparse
import json
import math
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import inject_missing_modalities, split_assay_cold, split_target_cold, split_temporal  # noqa: E402
from mirage_mini.experiment import run_model_suite  # noqa: E402
from mirage_mini.features import CachedTransformerEmbedder  # noqa: E402


PRIMARY_MODEL = "mirage_full"
MODELS = [
    PRIMARY_MODEL,
    "mirage_w_o_gate",
    "mirage_w_o_probe",
    "mirage_w_o_anchor",
    "hybrid_plus_pretrained_smiles_text",
    "hybrid_blend_avg",
    "historical_retrieval_evidence",
    "retrieval",
    "mask",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate absolute potency thresholds on one frozen CHEMBL_ASSAY benchmark.")
    parser.add_argument("--benchmark-dir", default="outputs/chembl_assay_revision_20260729")
    parser.add_argument("--output-root", default="outputs/revision_e05_absolute_thresholds_20260729")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[100.0, 300.0, 1000.0, 10000.0])
    parser.add_argument("--splits", nargs="+", default=["assay_cold", "target_cold", "temporal"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--missing-sequence-prob", type=float, default=0.35)
    parser.add_argument("--missing-text-prob", type=float, default=0.35)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--retrieval-reference-size", type=int, default=None)
    parser.add_argument("--pretrained-smiles-model", default="/home/test/wsk/hf_models/chemberta_zinc_base_v1")
    parser.add_argument("--pretrained-text-model", default="/home/test/wsk/hf_models/all_minilm_l6_v2")
    parser.add_argument("--pretrained-text-max-length", type=int, default=256)
    parser.add_argument("--pretrained-sequence-model", default="/home/test/wsk/hf_models/esm2_t6_8m_ur50d")
    parser.add_argument("--pretrained-sequence-max-length", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def pchembl_threshold(threshold_nm: float) -> float:
    if threshold_nm <= 0:
        raise ValueError("threshold_nm must be positive")
    return float(9.0 - math.log10(float(threshold_nm)))


def label_from_threshold(frame: pd.DataFrame, threshold_nm: float) -> pd.DataFrame:
    out = frame.copy()
    values = pd.to_numeric(out["pchembl_value"], errors="coerce")
    if values.isna().any():
        raise ValueError("pchembl_value must be complete before threshold evaluation")
    out["label"] = (values >= pchembl_threshold(threshold_nm)).astype(int)
    out["binary_label_policy"] = f"absolute pChEMBL >= {pchembl_threshold(threshold_nm):.4f} (<= {threshold_nm:g} nM)"
    return out


def split_frame(frame: pd.DataFrame, split_mode: str, seed: int) -> dict[str, pd.DataFrame]:
    if split_mode == "assay_cold":
        return split_assay_cold(frame, seed=seed)
    if split_mode == "target_cold":
        return split_target_cold(frame, seed=seed)
    if split_mode == "temporal":
        return split_temporal(frame, seed=seed)
    raise ValueError(f"Unsupported split: {split_mode}")


def build_embedder(model_name: str, cache_dir: Path, max_length: int, batch_size: int = 64):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower() or "model"
    return CachedTransformerEmbedder(
        model_name=model_name,
        cache_path=cache_dir / "embedding_cache" / f"{slug}.pkl",
        batch_size=batch_size,
        max_length=max_length,
    )


def extra_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (np.asarray(probability) >= 0.5).astype(int)
    return {
        "brier": float(brier_score_loss(y_true, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
    }


def summarize_run(run_dir: Path) -> list[dict]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(run_dir / "predictions.csv")
    y_true = predictions["label"].to_numpy(dtype=int)
    rows: list[dict] = []
    for model in MODELS:
        payload = metrics.get("models", {}).get(model)
        if not isinstance(payload, dict):
            continue
        for metric_split, suffix in [("test_clean", "clean"), ("test_missing", "missing")]:
            metric = payload.get(metric_split)
            if not isinstance(metric, dict):
                continue
            column = f"{model}_prob_{suffix}"
            if column not in predictions:
                continue
            probability = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
            rows.append(
                {
                    "run_dir": run_dir.name,
                    "dataset": "CHEMBL_ASSAY",
                    "threshold_nm": metrics["threshold_nm"],
                    "pchembl_threshold": metrics["pchembl_threshold"],
                    "split_mode": metrics["split_mode"],
                    "seed": metrics["seed"],
                    "metric_split": metric_split,
                    "model": model,
                    "n_train": metrics["split_sizes"]["train"],
                    "n_val": metrics["split_sizes"]["val"],
                    "n_test": int(len(predictions)),
                    "test_prevalence": metrics["split_prevalence"]["test"],
                    "auprc": metric.get("auprc"),
                    "auprc_lift": float(metric["auprc"] - metrics["split_prevalence"]["test"]),
                    "auroc": metric.get("auroc"),
                    "ece": metric.get("ece"),
                    "risk_at_80_coverage": metric.get("risk_at_80_coverage"),
                    **extra_metrics(y_true, probability),
                }
            )
    return rows


def run_one(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
    threshold_nm: float,
    split_mode: str,
    seed: int,
    args: argparse.Namespace,
    smiles_embedder,
    text_embedder,
    sequence_embedder,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics.json").exists() and (output_dir / "predictions.csv").exists() and not args.overwrite:
        return summarize_run(output_dir)
    labeled = label_from_threshold(frame, threshold_nm)
    splits = split_frame(labeled, split_mode, seed)
    for name, split in splits.items():
        if split["label"].nunique() < 2:
            raise ValueError(f"{split_mode} seed={seed} threshold={threshold_nm:g} has one class in {name}")
    stressed = inject_missing_modalities(
        splits["test"],
        probs={"sequence": args.missing_sequence_prob, "text": args.missing_text_prob},
        seed=seed + 1,
    )
    suite = run_model_suite(
        train_df=splits["train"],
        val_df=splits["val"],
        test_df=splits["test"],
        stressed_test_df=stressed,
        n_neighbors=args.n_neighbors,
        smiles_embedder=smiles_embedder,
        text_embedder=text_embedder,
        sequence_embedder=sequence_embedder,
        retrieval_reference_size=args.retrieval_reference_size,
    )
    predictions = suite.pop("predictions")
    validation_predictions = suite.pop("validation_predictions")
    metrics = {
        "dataset": "CHEMBL_ASSAY",
        "primary_model": PRIMARY_MODEL,
        "threshold_nm": float(threshold_nm),
        "pchembl_threshold": pchembl_threshold(threshold_nm),
        "binary_label_policy": labeled["binary_label_policy"].iloc[0],
        "split_mode": split_mode,
        "seed": int(seed),
        "split_sizes": {name: int(len(value)) for name, value in splits.items()},
        "split_prevalence": {name: float(value["label"].mean()) for name, value in splits.items()},
        "missing_sequence_prob": args.missing_sequence_prob,
        "missing_text_prob": args.missing_text_prob,
        "n_neighbors": args.n_neighbors,
        "retrieval_reference_size": args.retrieval_reference_size,
        "models": suite["models"],
    }
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    splits["train"].to_csv(output_dir / "train_preview.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return summarize_run(output_dir)


def main() -> None:
    args = parse_args()
    cache_dir = (REPO_ROOT / args.cache_dir).resolve()
    frame = pd.read_csv((REPO_ROOT / args.benchmark_dir).resolve() / "benchmark_frame.csv")
    output_root = (REPO_ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    smiles_embedder = build_embedder(args.pretrained_smiles_model, cache_dir, max_length=128)
    text_embedder = build_embedder(args.pretrained_text_model, cache_dir, max_length=args.pretrained_text_max_length)
    sequence_embedder = build_embedder(args.pretrained_sequence_model, cache_dir, max_length=args.pretrained_sequence_max_length, batch_size=16)

    rows: list[dict] = []
    failures: list[dict] = []
    for threshold in args.thresholds:
        for split_mode in args.splits:
            for seed in args.seeds:
                run_dir = output_root / f"{split_mode}_t{threshold:g}nm_s{seed}"
                try:
                    print(f"[threshold] START {run_dir.name}", flush=True)
                    rows.extend(run_one(
                        frame=frame, output_dir=run_dir, threshold_nm=threshold, split_mode=split_mode,
                        seed=seed, args=args, smiles_embedder=smiles_embedder, text_embedder=text_embedder,
                        sequence_embedder=sequence_embedder,
                    ))
                    print(f"[threshold] DONE {run_dir.name}", flush=True)
                except Exception as exc:
                    failures.append({"threshold_nm": threshold, "split_mode": split_mode, "seed": seed, "error": str(exc), "traceback": traceback.format_exc()})
                    print(f"[threshold] FAIL {run_dir.name}: {exc}", flush=True)
    long = pd.DataFrame(rows)
    long.to_csv(output_root / "threshold_sensitivity_long.csv", index=False)
    if not long.empty:
        summary = long.groupby(["threshold_nm", "pchembl_threshold", "split_mode", "metric_split", "model"], as_index=False).agg(
            runs=("seed", "nunique"), n_test=("n_test", "mean"), test_prevalence=("test_prevalence", "mean"),
            auprc_mean=("auprc", "mean"), auprc_std=("auprc", "std"), auprc_lift_mean=("auprc_lift", "mean"),
            auroc_mean=("auroc", "mean"), auroc_std=("auroc", "std"), ece_mean=("ece", "mean"),
            risk_at_80_mean=("risk_at_80_coverage", "mean"), brier_mean=("brier", "mean"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"), mcc_mean=("mcc", "mean"),
        )
        summary.to_csv(output_root / "threshold_sensitivity_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(output_root / "threshold_sensitivity_failures.csv", index=False)
    (output_root / "manifest.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
