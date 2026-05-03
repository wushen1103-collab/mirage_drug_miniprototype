from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.experiment import run_single_tdc_official_experiment
from mirage_mini.features import CachedTransformerEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BindingDB_Kd")
    parser.add_argument(
        "--split-method",
        default="cold_target",
        choices=["random", "cold_drug", "cold_target", "cold_pair"],
    )
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", default="outputs/tdc_official_run01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-sequence-prob", type=float, default=0.35)
    parser.add_argument("--missing-text-prob", type=float, default=0.35)
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    parser.add_argument("--inactive-threshold-nm", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--retrieval-reference-size", type=int, default=None)
    parser.add_argument("--enable-pretrained-smiles", action="store_true")
    parser.add_argument("--pretrained-smiles-model", default="seyonec/ChemBERTa-zinc-base-v1")
    parser.add_argument("--pretrained-smiles-batch-size", type=int, default=64)
    parser.add_argument("--pretrained-smiles-max-length", type=int, default=128)
    parser.add_argument("--enable-pretrained-text", action="store_true")
    parser.add_argument("--pretrained-text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--pretrained-text-batch-size", type=int, default=64)
    parser.add_argument("--pretrained-text-max-length", type=int, default=128)
    parser.add_argument("--enable-pretrained-sequence", action="store_true")
    parser.add_argument("--pretrained-sequence-model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--pretrained-sequence-batch-size", type=int, default=32)
    parser.add_argument("--pretrained-sequence-max-length", type=int, default=512)
    parser.add_argument("--enable-interaction-probe", action="store_true")
    parser.add_argument("--interaction-probe-device", default=None)
    parser.add_argument("--interaction-probe-text-field", default="text")
    parser.add_argument("--interaction-probe-batch-size", type=int, default=64)
    parser.add_argument("--interaction-probe-max-epochs", type=int, default=60)
    parser.add_argument("--interaction-probe-patience", type=int, default=8)
    parser.add_argument("--interaction-probe-hidden-dim", type=int, default=256)
    parser.add_argument("--interaction-probe-proj-dim", type=int, default=128)
    parser.add_argument("--interaction-probe-dropout", type=float, default=0.15)
    parser.add_argument("--interaction-probe-lr", type=float, default=3e-4)
    parser.add_argument("--interaction-probe-weight-decay", type=float, default=3e-4)
    parser.add_argument("--interaction-probe-text-dropout-prob", type=float, default=0.15)
    return parser.parse_args()


def _build_embedder(
    enabled: bool,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    max_length: int,
):
    if not enabled:
        return None
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower() or "model"
    return CachedTransformerEmbedder(
        model_name=model_name,
        cache_path=cache_dir / "embedding_cache" / f"{model_slug}.pkl",
        batch_size=batch_size,
        max_length=max_length,
    )


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_embedder = _build_embedder(
        enabled=args.enable_pretrained_smiles,
        model_name=args.pretrained_smiles_model,
        cache_dir=cache_dir,
        batch_size=args.pretrained_smiles_batch_size,
        max_length=args.pretrained_smiles_max_length,
    )
    text_embedder = _build_embedder(
        enabled=args.enable_pretrained_text,
        model_name=args.pretrained_text_model,
        cache_dir=cache_dir,
        batch_size=args.pretrained_text_batch_size,
        max_length=args.pretrained_text_max_length,
    )
    sequence_embedder = _build_embedder(
        enabled=args.enable_pretrained_sequence,
        model_name=args.pretrained_sequence_model,
        cache_dir=cache_dir,
        batch_size=args.pretrained_sequence_batch_size,
        max_length=args.pretrained_sequence_max_length,
    )

    result = run_single_tdc_official_experiment(
        dataset=args.dataset,
        cache_dir=cache_dir,
        seed=args.seed,
        split_method=args.split_method,
        missing_sequence_prob=args.missing_sequence_prob,
        missing_text_prob=args.missing_text_prob,
        activity_threshold_nm=args.activity_threshold_nm,
        inactive_threshold_nm=args.inactive_threshold_nm,
        n_neighbors=args.n_neighbors,
        retrieval_reference_size=args.retrieval_reference_size,
        smiles_embedder=smiles_embedder,
        text_embedder=text_embedder,
        sequence_embedder=sequence_embedder,
        enable_interaction_probe=args.enable_interaction_probe,
        interaction_probe_config={
            "device": args.interaction_probe_device,
            "text_field": args.interaction_probe_text_field,
            "batch_size": args.interaction_probe_batch_size,
            "max_epochs": args.interaction_probe_max_epochs,
            "patience": args.interaction_probe_patience,
            "hidden_dim": args.interaction_probe_hidden_dim,
            "proj_dim": args.interaction_probe_proj_dim,
            "dropout": args.interaction_probe_dropout,
            "lr": args.interaction_probe_lr,
            "weight_decay": args.interaction_probe_weight_decay,
            "text_dropout_prob": args.interaction_probe_text_dropout_prob,
            "seed": args.seed,
        }
        if args.enable_interaction_probe
        else None,
    )
    predictions = result.pop("predictions")
    validation_predictions = result.pop("validation_predictions")
    train_preview = result.pop("train_preview")

    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    train_preview.to_csv(output_dir / "train_preview.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

