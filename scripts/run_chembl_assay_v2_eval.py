from __future__ import annotations

import argparse
import json
import re
import sys
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

# The first entry is the fixed MIRAGE-DTA architecture. The remaining entries
# are pre-specified component and fusion controls, not per-split selections.
MAIN_MODELS = [
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", default="outputs/chembl_assay_revision_20260729")
    parser.add_argument("--output-dir", default="outputs/chembl_assay_revision_eval_20260729")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--splits", nargs="+", default=["assay_cold", "target_cold", "temporal"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--missing-sequence-prob", type=float, default=0.35)
    parser.add_argument("--missing-text-prob", type=float, default=0.35)
    parser.add_argument("--n-neighbors", type=int, default=1)
    parser.add_argument("--retrieval-reference-size", type=int, default=5000)
    parser.add_argument("--pretrained-smiles-model", default="/home/test/wsk/hf_models/chemberta_zinc_base_v1")
    parser.add_argument("--pretrained-text-model", default="/home/test/wsk/hf_models/all_minilm_l6_v2")
    parser.add_argument("--pretrained-text-max-length", type=int, default=256)
    parser.add_argument("--pretrained-sequence-model", default="/home/test/wsk/hf_models/esm2_t6_8m_ur50d")
    parser.add_argument("--pretrained-sequence-max-length", type=int, default=512)
    parser.add_argument("--interaction-probe-device", default=None)
    parser.add_argument("--interaction-probe-batch-size", type=int, default=64)
    parser.add_argument("--interaction-probe-max-epochs", type=int, default=60)
    parser.add_argument("--interaction-probe-patience", type=int, default=8)
    parser.add_argument("--interaction-probe-hidden-dim", type=int, default=256)
    parser.add_argument("--interaction-probe-proj-dim", type=int, default=128)
    parser.add_argument("--interaction-probe-dropout", type=float, default=0.15)
    parser.add_argument("--interaction-probe-lr", type=float, default=3e-4)
    parser.add_argument("--interaction-probe-weight-decay", type=float, default=3e-4)
    parser.add_argument("--interaction-probe-text-dropout-prob", type=float, default=0.15)
    parser.add_argument("--enable-interaction-probe", dest="enable_interaction_probe", action="store_true")
    parser.add_argument("--disable-interaction-probe", dest="enable_interaction_probe", action="store_false")
    parser.set_defaults(enable_interaction_probe=False)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _split_frame(frame: pd.DataFrame, split_mode: str, seed: int) -> dict[str, pd.DataFrame]:
    if split_mode == "assay_cold":
        return split_assay_cold(frame, seed=seed)
    if split_mode == "target_cold":
        return split_target_cold(frame, seed=seed)
    if split_mode == "temporal":
        return split_temporal(frame, seed=seed)
    raise ValueError(f"Unsupported split: {split_mode}")


def _build_embedder(model_name: str, cache_dir: Path, max_length: int, batch_size: int = 64):
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower() or "model"
    return CachedTransformerEmbedder(
        model_name=model_name,
        cache_path=cache_dir / "embedding_cache" / f"{model_slug}.pkl",
        batch_size=batch_size,
        max_length=max_length,
    )


def _extra_binary(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    return {
        "brier": float(brier_score_loss(y_true, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def _summary_from_existing(output_dir: Path) -> list[dict]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    pred = pd.read_csv(output_dir / "predictions.csv")
    rows: list[dict] = []
    y_true = pred["label"].to_numpy(dtype=int)
    for model in MAIN_MODELS:
        payload = metrics.get("models", {}).get(model)
        if not isinstance(payload, dict):
            continue
        for metric_split, suffix in [("test_clean", "clean"), ("test_missing", "missing")]:
            split_payload = payload.get(metric_split)
            if not isinstance(split_payload, dict):
                continue
            prob_col = f"{model}_prob_{suffix}"
            extra = {}
            if prob_col in pred.columns:
                extra = _extra_binary(y_true, pd.to_numeric(pred[prob_col], errors="coerce").to_numpy(dtype=float))
            prevalence = metrics.get("split_prevalence", {}).get("test")
            auprc = split_payload.get("auprc")
            rows.append(
                {
                    "run_dir": output_dir.name,
                    "dataset": "CHEMBL_ASSAY",
                    "split_mode": metrics.get("split_mode"),
                    "seed": metrics.get("seed"),
                    "metric_split": metric_split,
                    "model": model,
                    "n_train": metrics.get("split_sizes", {}).get("train"),
                    "n_val": metrics.get("split_sizes", {}).get("val"),
                    "n_test": int(len(pred)),
                    "test_prevalence": prevalence,
                    "auprc": auprc,
                    "auprc_lift": float(auprc - prevalence) if auprc is not None and prevalence is not None else np.nan,
                    "auroc": split_payload.get("auroc"),
                    "ece": split_payload.get("ece"),
                    "risk_at_80_coverage": split_payload.get("risk_at_80_coverage"),
                    **extra,
                }
            )
    return rows


def _run_one(
    *,
    frame: pd.DataFrame,
    output_dir: Path,
    split_mode: str,
    seed: int,
    args: argparse.Namespace,
    smiles_embedder,
    text_embedder,
    sequence_embedder,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics.json").exists() and (output_dir / "predictions.csv").exists() and not args.overwrite:
        return _summary_from_existing(output_dir)

    splits = _split_frame(frame, split_mode=split_mode, seed=seed)
    for name, split_df in splits.items():
        if split_df["label"].nunique() < 2:
            raise ValueError(f"{split_mode} seed={seed} has one class in {name}")
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
        enable_interaction_probe=args.enable_interaction_probe,
        interaction_probe_config={
            "device": args.interaction_probe_device,
            "batch_size": args.interaction_probe_batch_size,
            "max_epochs": args.interaction_probe_max_epochs,
            "patience": args.interaction_probe_patience,
            "hidden_dim": args.interaction_probe_hidden_dim,
            "proj_dim": args.interaction_probe_proj_dim,
            "dropout": args.interaction_probe_dropout,
            "lr": args.interaction_probe_lr,
            "weight_decay": args.interaction_probe_weight_decay,
            "text_dropout_prob": args.interaction_probe_text_dropout_prob,
            "seed": seed,
        },
        retrieval_reference_size=args.retrieval_reference_size,
    )
    predictions = suite.pop("predictions")
    validation_predictions = suite.pop("validation_predictions")
    metrics = {
        "dataset": "CHEMBL_ASSAY",
        "primary_model": PRIMARY_MODEL,
        "split_mode": split_mode,
        "seed": int(seed),
        "split_sizes": {name: int(len(split_df)) for name, split_df in splits.items()},
        "split_prevalence": {name: float(split_df["label"].mean()) for name, split_df in splits.items()},
        "missing_sequence_prob": args.missing_sequence_prob,
        "missing_text_prob": args.missing_text_prob,
        "n_neighbors": int(args.n_neighbors),
        "retrieval_reference_size": int(args.retrieval_reference_size),
        "models": suite["models"],
    }
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    splits["train"].to_csv(output_dir / "train_preview.csv", index=False)
    splits["val"].to_csv(output_dir / "val_preview.csv", index=False)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=_json_default), encoding="utf-8")
    return _summary_from_existing(output_dir)


def main() -> None:
    args = parse_args()
    cache_dir = (REPO_ROOT / args.cache_dir).resolve()
    benchmark_dir = (REPO_ROOT / args.benchmark_dir).resolve()
    output_root = (REPO_ROOT / args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(benchmark_dir / "benchmark_frame.csv")

    smiles_embedder = _build_embedder(args.pretrained_smiles_model, cache_dir=cache_dir, max_length=128)
    text_embedder = _build_embedder(
        args.pretrained_text_model,
        cache_dir=cache_dir,
        max_length=args.pretrained_text_max_length,
    )
    sequence_embedder = _build_embedder(
        args.pretrained_sequence_model,
        cache_dir=cache_dir,
        max_length=args.pretrained_sequence_max_length,
        batch_size=16,
    )

    all_rows: list[dict] = []
    for split_mode in args.splits:
        for seed in args.seeds:
            run_dir = output_root / f"{split_mode}_s{seed}"
            print(f"[v2-eval] running {split_mode} seed={seed}", flush=True)
            rows = _run_one(
                frame=frame,
                output_dir=run_dir,
                split_mode=split_mode,
                seed=seed,
                args=args,
                smiles_embedder=smiles_embedder,
                text_embedder=text_embedder,
                sequence_embedder=sequence_embedder,
            )
            all_rows.extend(rows)
            pd.DataFrame(all_rows).to_csv(output_root / "all_runs.csv", index=False)

    all_df = pd.DataFrame(all_rows)
    summary = (
        all_df.groupby(["dataset", "split_mode", "metric_split", "model"], as_index=False)
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
        .sort_values(["split_mode", "metric_split", "model"])
    )
    summary.to_csv(output_root / "summary.csv", index=False)
    manifest = {
        "benchmark_dir": str(benchmark_dir),
        "output_dir": str(output_root),
        "splits": args.splits,
        "seeds": args.seeds,
        "missing_sequence_prob": args.missing_sequence_prob,
        "missing_text_prob": args.missing_text_prob,
        "n_neighbors": args.n_neighbors,
        "retrieval_reference_size": args.retrieval_reference_size,
        "primary_model": PRIMARY_MODEL,
        "models": MAIN_MODELS,
        "pretrained_smiles_model": args.pretrained_smiles_model,
        "pretrained_text_model": args.pretrained_text_model,
        "pretrained_sequence_model": args.pretrained_sequence_model,
        "enable_interaction_probe": args.enable_interaction_probe,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

