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

import pandas as pd

from mirage_mini.data import inject_missing_modalities
from mirage_mini.experiment import load_bundle, make_splits, run_model_suite
from mirage_mini.features import CachedTransformerEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="CHEMBL_ASSAY")
    parser.add_argument("--sample-size", type=int, default=250)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", default="outputs/matrix01")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--split-modes", default="target_cold,assay_cold,random")
    parser.add_argument("--missing-settings", default="0.0:0.0,0.35:0.35,0.6:0.6")
    parser.add_argument("--n-neighbors", type=int, default=8)
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    seeds = [int(x) for x in args.seeds.split(",") if x]
    split_modes = [x.strip() for x in args.split_modes.split(",") if x.strip()]
    missing_settings = []
    for item in args.missing_settings.split(","):
        if not item.strip():
            continue
        seq_prob, text_prob = item.split(":")
        missing_settings.append((float(seq_prob), float(text_prob)))

    smiles_embedder = None
    if args.enable_pretrained_smiles:
        model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.pretrained_smiles_model).strip("_").lower() or "model"
        smiles_embedder = CachedTransformerEmbedder(
            model_name=args.pretrained_smiles_model,
            cache_path=cache_dir / "embedding_cache" / f"{model_slug}.pkl",
            batch_size=args.pretrained_smiles_batch_size,
            max_length=args.pretrained_smiles_max_length,
        )
    text_embedder = None
    if args.enable_pretrained_text:
        model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.pretrained_text_model).strip("_").lower() or "model"
        text_embedder = CachedTransformerEmbedder(
            model_name=args.pretrained_text_model,
            cache_path=cache_dir / "embedding_cache" / f"{model_slug}.pkl",
            batch_size=args.pretrained_text_batch_size,
            max_length=args.pretrained_text_max_length,
        )
    sequence_embedder = None
    if args.enable_pretrained_sequence:
        model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", args.pretrained_sequence_model).strip("_").lower() or "model"
        sequence_embedder = CachedTransformerEmbedder(
            model_name=args.pretrained_sequence_model,
            cache_path=cache_dir / "embedding_cache" / f"{model_slug}.pkl",
            batch_size=args.pretrained_sequence_batch_size,
            max_length=args.pretrained_sequence_max_length,
        )

    rows = []
    for seed in seeds:
        bundle = load_bundle(
            dataset=args.dataset,
            cache_dir=cache_dir,
            sample_size=args.sample_size,
            seed=seed,
        )
        for split_mode in split_modes:
            splits = make_splits(bundle.frame, split_mode=split_mode, seed=seed)
            for seq_prob, text_prob in missing_settings:
                stressed_test_df = inject_missing_modalities(
                    splits["test"],
                    probs={"sequence": seq_prob, "text": text_prob},
                    seed=seed + 1,
                )
                suite = run_model_suite(
                    train_df=splits["train"],
                    val_df=splits["val"],
                    test_df=splits["test"],
                    stressed_test_df=stressed_test_df,
                    n_neighbors=args.n_neighbors,
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
                        "seed": seed,
                    }
                    if args.enable_interaction_probe
                    else None,
                )
                for model_name, model_metrics in suite["models"].items():
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "seed": seed,
                            "split_mode": split_mode,
                            "missing_sequence_prob": seq_prob,
                            "missing_text_prob": text_prob,
                            "model": model_name,
                            "test_clean_auroc": model_metrics["test_clean"]["auroc"],
                            "test_clean_auprc": model_metrics["test_clean"]["auprc"],
                            "test_missing_auroc": model_metrics["test_missing"]["auroc"],
                            "test_missing_auprc": model_metrics["test_missing"]["auprc"],
                            "test_missing_ece": model_metrics["test_missing"]["ece"],
                            "test_missing_risk80": model_metrics["test_missing"]["risk_at_80_coverage"],
                        }
                    )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["dataset", "split_mode", "missing_sequence_prob", "missing_text_prob", "model"])
        .agg(
            mean_test_missing_auroc=("test_missing_auroc", "mean"),
            std_test_missing_auroc=("test_missing_auroc", "std"),
            mean_test_missing_auprc=("test_missing_auprc", "mean"),
            std_test_missing_auprc=("test_missing_auprc", "std"),
            mean_test_missing_ece=("test_missing_ece", "mean"),
            mean_test_missing_risk80=("test_missing_risk80", "mean"),
        )
        .reset_index()
    )
    frame.to_csv(output_dir / "all_runs.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "matrix_config.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "sample_size": args.sample_size,
                "seeds": seeds,
                "split_modes": split_modes,
                "missing_settings": missing_settings,
                "n_neighbors": args.n_neighbors,
                "enable_pretrained_smiles": args.enable_pretrained_smiles,
                "pretrained_smiles_model": args.pretrained_smiles_model if args.enable_pretrained_smiles else None,
                "pretrained_smiles_batch_size": args.pretrained_smiles_batch_size if args.enable_pretrained_smiles else None,
                "pretrained_smiles_max_length": args.pretrained_smiles_max_length if args.enable_pretrained_smiles else None,
                "enable_pretrained_text": args.enable_pretrained_text,
                "pretrained_text_model": args.pretrained_text_model if args.enable_pretrained_text else None,
                "pretrained_text_batch_size": args.pretrained_text_batch_size if args.enable_pretrained_text else None,
                "pretrained_text_max_length": args.pretrained_text_max_length if args.enable_pretrained_text else None,
                "enable_pretrained_sequence": args.enable_pretrained_sequence,
                "pretrained_sequence_model": args.pretrained_sequence_model if args.enable_pretrained_sequence else None,
                "pretrained_sequence_batch_size": args.pretrained_sequence_batch_size if args.enable_pretrained_sequence else None,
                "pretrained_sequence_max_length": args.pretrained_sequence_max_length if args.enable_pretrained_sequence else None,
                "enable_interaction_probe": args.enable_interaction_probe,
                "interaction_probe_device": args.interaction_probe_device if args.enable_interaction_probe else None,
                "interaction_probe_text_field": args.interaction_probe_text_field if args.enable_interaction_probe else None,
                "interaction_probe_batch_size": args.interaction_probe_batch_size if args.enable_interaction_probe else None,
                "interaction_probe_max_epochs": args.interaction_probe_max_epochs if args.enable_interaction_probe else None,
                "interaction_probe_patience": args.interaction_probe_patience if args.enable_interaction_probe else None,
                "interaction_probe_hidden_dim": args.interaction_probe_hidden_dim if args.enable_interaction_probe else None,
                "interaction_probe_proj_dim": args.interaction_probe_proj_dim if args.enable_interaction_probe else None,
                "interaction_probe_dropout": args.interaction_probe_dropout if args.enable_interaction_probe else None,
                "interaction_probe_lr": args.interaction_probe_lr if args.enable_interaction_probe else None,
                "interaction_probe_weight_decay": args.interaction_probe_weight_decay if args.enable_interaction_probe else None,
                "interaction_probe_text_dropout_prob": args.interaction_probe_text_dropout_prob if args.enable_interaction_probe else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

