from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import prepare_benchmark_splits
from mirage_mini.external_baselines import (
    compute_ranking_metrics,
    prepare_external_regression_frame,
    save_external_metrics,
    subsample_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balm-repo", required=True)
    parser.add_argument("--dataset", default="BindingDB_Kd")
    parser.add_argument("--split-method", default="cold_target", choices=["random", "cold_drug", "cold_target", "cold_pair", "target_cold", "assay_cold", "temporal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-variant", default="projection", choices=["baseline", "projection", "peft"])
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    parser.add_argument("--inactive-threshold-nm", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--protein-model", default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--drug-model", default="DeepChem/ChemBERTa-77M-MTR")
    parser.add_argument("--protein-max-length", type=int, default=1024)
    parser.add_argument("--drug-max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--projected-size", type=int, default=256)
    parser.add_argument("--projected-dropout", type=float, default=0.3)
    parser.add_argument("--warmup-steps-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def _import_balm(balm_repo: Path):
    if str(balm_repo) not in sys.path:
        sys.path.insert(0, str(balm_repo))
    from balm.configs import Configs
    from balm.datasets.utils import DataCollatorWithPadding
    from balm.tokenization import pre_tokenize_unique_entities, tokenize_with_lookup
    from balm.trainer import Trainer
    from datasets import Dataset

    return Configs, DataCollatorWithPadding, Dataset, Trainer, pre_tokenize_unique_entities, tokenize_with_lookup


def _build_config_dict(args: argparse.Namespace) -> dict:
    if args.model_variant == "baseline":
        protein_type = "baseline"
        drug_type = "baseline"
        protein_peft = None
        drug_peft = None
        loss_function = "baseline_mse"
        epochs = max(args.epochs, 150)
    elif args.model_variant == "projection":
        protein_type = "projection"
        drug_type = "projection"
        protein_peft = None
        drug_peft = None
        loss_function = "cosine_mse"
        epochs = args.epochs
    else:
        protein_type = "lokr"
        drug_type = "loha"
        protein_peft = {
            "r": 16,
            "alpha": 32,
            "rank_dropout": 0.0,
            "module_dropout": 0.0,
            "target_modules": ["key", "query", "value"],
        }
        drug_peft = {
            "r": 16,
            "alpha": 32,
            "rank_dropout": 0.0,
            "module_dropout": 0.0,
            "target_modules": ["key", "query", "value"],
        }
        loss_function = "cosine_mse"
        epochs = args.epochs

    return {
        "model_configs": {
            "protein_model_name_or_path": args.protein_model,
            "drug_model_name_or_path": args.drug_model,
            "model_hyperparameters": {
                "learning_rate": args.learning_rate,
                "warmup_steps_ratio": args.warmup_steps_ratio,
                "protein_max_seq_len": args.protein_max_length,
                "drug_max_seq_len": args.drug_max_length,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "projected_size": args.projected_size,
                "projected_dropout": args.projected_dropout,
            },
            "protein_fine_tuning_type": protein_type,
            "protein_peft_hyperparameters": protein_peft,
            "drug_fine_tuning_type": drug_type,
            "drug_peft_hyperparameters": drug_peft,
            "loss_function": loss_function,
        },
        "dataset_configs": {
            "dataset_name": f"DTI_{args.dataset}",
            "split_method": args.split_method,
            "harmonize_affinities_mode": "max_affinity",
        },
        "training_configs": {
            "random_seed": args.seed,
            "device": 0,
            "epochs": epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "outputs_dir": str(args.output_dir),
        },
    }


def _prepare_split_frames(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    splits = prepare_benchmark_splits(
        dataset_name=args.dataset,
        cache_dir=Path(args.cache_dir),
        split_method=args.split_method,
        seed=args.seed,
        activity_threshold_nm=args.activity_threshold_nm,
        inactive_threshold_nm=args.inactive_threshold_nm,
        sample_size=args.sample_size,
    )
    external_splits = {
        name: prepare_external_regression_frame(frame, dataset_name=args.dataset)
        for name, frame in splits.items()
    }
    return {
        "train": subsample_frame(external_splits["train"], args.max_train_samples, args.seed),
        "val": subsample_frame(external_splits["val"], args.max_val_samples, args.seed + 1),
        "test": subsample_frame(external_splits["test"], args.max_test_samples, args.seed + 2),
    }


def _attach_custom_splits(
    trainer,
    split_frames: dict[str, pd.DataFrame],
    *,
    Dataset,
    DataCollatorWithPadding,
    pre_tokenize_unique_entities,
    tokenize_with_lookup,
) -> dict[str, pd.DataFrame]:
    trainer.pkd_lower_bound = float(split_frames["train"]["Y"].min())
    trainer.pkd_upper_bound = float(split_frames["train"]["Y"].max())

    filtered_raw_frames: dict[str, pd.DataFrame] = {}
    for split_name, split_df in split_frames.items():
        raw_frame = split_df.reset_index(drop=True).copy()
        work = raw_frame.copy()
        if trainer.model_configs.loss_function == "cosine_mse":
            pkd_range = trainer.pkd_upper_bound - trainer.pkd_lower_bound
            if pkd_range == 0:
                work["Y"] = 0.0
            else:
                work["Y"] = (work["Y"] - trainer.pkd_lower_bound) / pkd_range * 2.0 - 1.0

        collator = DataCollatorWithPadding(
            protein_tokenizer=trainer.protein_tokenizer,
            drug_tokenizer=trainer.drug_tokenizer,
            padding="max_length",
            protein_max_length=trainer.protein_max_seq_len,
            drug_max_length=trainer.drug_max_seq_len,
            return_tensors="pt",
        )

        balm_frame = work[["sample_id", "drug_id", "target_id", "Drug", "Target", "Y"]].copy()
        balm_frame["row_id"] = np.arange(len(work))
        protein_lookup, drug_lookup = pre_tokenize_unique_entities(
            balm_frame,
            trainer.protein_tokenizer,
            trainer.drug_tokenizer,
        )
        dataset = Dataset.from_pandas(balm_frame, preserve_index=False).map(
            lambda example: tokenize_with_lookup(example, protein_lookup, drug_lookup)
        )
        dataset = dataset.filter(
            lambda example: len(example["protein_input_ids"]) <= trainer.protein_max_seq_len
            and len(example["drug_input_ids"]) <= trainer.drug_max_seq_len
        )
        kept_row_ids = dataset["row_id"]
        filtered_raw_frames[split_name] = raw_frame.iloc[kept_row_ids].reset_index(drop=True)
        dataset = dataset.remove_columns(["row_id"])
        dataloader = DataLoader(
            dataset,
            shuffle=(split_name == "train"),
            collate_fn=collator,
            batch_size=trainer.training_configs.batch_size,
            pin_memory=True,
        )
        if split_name == "train":
            trainer.train_dataloader = dataloader
        elif split_name == "val":
            trainer.valid_dataloader = dataloader
        elif split_name == "test":
            trainer.test_dataloader = dataloader

    return filtered_raw_frames


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT_NAME", "mirage_external_baselines")
    os.environ.setdefault("WANDB_ENTITY", "")

    (
        Configs,
        DataCollatorWithPadding,
        Dataset,
        Trainer,
        pre_tokenize_unique_entities,
        tokenize_with_lookup,
    ) = _import_balm(Path(args.balm_repo).resolve())

    config_dict = _build_config_dict(args)
    configs = Configs(**config_dict)
    trainer = Trainer(
        configs=configs,
        wandb_entity=os.environ.get("WANDB_ENTITY", ""),
        wandb_project=os.environ.get("WANDB_PROJECT_NAME", "mirage_external_baselines"),
        outputs_dir=str(output_dir),
    )

    split_frames = _prepare_split_frames(args)
    for split_name, split_df in split_frames.items():
        split_df.to_csv(output_dir / f"{split_name}_official_split.csv", index=False)

    filtered_frames = _attach_custom_splits(
        trainer,
        split_frames,
        Dataset=Dataset,
        DataCollatorWithPadding=DataCollatorWithPadding,
        pre_tokenize_unique_entities=pre_tokenize_unique_entities,
        tokenize_with_lookup=tokenize_with_lookup,
    )

    trainer.setup_training()
    trainer.train()
    regression_metrics = trainer.test("test", save_prediction=True)

    pred_path = output_dir / "test_prediction.csv"
    pred_df = pd.read_csv(pred_path)
    if "Unnamed: 0" in pred_df.columns:
        pred_df = pred_df.drop(columns=["Unnamed: 0"])
    pred_df = pred_df.reset_index(drop=True)

    test_frame = filtered_frames["test"].reset_index(drop=True)
    if len(test_frame) != len(pred_df):
        raise ValueError(
            f"Prediction rows ({len(pred_df)}) do not match test rows ({len(test_frame)})."
        )

    merged = test_frame[
        ["sample_id", "drug_id", "target_id", "Drug", "Target", "binary_label", "label_raw_nm", "Y"]
    ].copy()
    merged["pkd_label"] = pred_df["label"]
    merged["pkd_prediction"] = pred_df["prediction"]
    merged.to_csv(output_dir / "test_prediction_with_binary.csv", index=False)

    ranking_metrics = compute_ranking_metrics(
        merged["binary_label"].to_numpy(),
        merged["pkd_prediction"].to_numpy(),
    )
    metrics = {
        **ranking_metrics,
        "rmse": float(regression_metrics["test/rmse"]),
        "pearson": float(regression_metrics["test/pearson"]),
        "spearman": float(regression_metrics["test/spearman"]),
        "ci": float(regression_metrics["test/ci"]),
    }
    metric_split = (
        "test_maxlen_filtered"
        if len(filtered_frames["test"]) != len(split_frames["test"])
        else "test_clean"
    )
    payload = save_external_metrics(
        output_dir=output_dir,
        framework="balm",
        model=f"balm_{args.model_variant}",
        dataset=args.dataset,
        split_mode=args.split_method,
        seed=args.seed,
        metric_split=metric_split,
        metrics=metrics,
        run_meta={
            "protein_model": args.protein_model,
            "drug_model": args.drug_model,
            "train_size_raw": int(len(split_frames["train"])),
            "val_size_raw": int(len(split_frames["val"])),
            "test_size_raw": int(len(split_frames["test"])),
            "train_size_used": int(len(filtered_frames["train"])),
            "val_size_used": int(len(filtered_frames["val"])),
            "test_size_used": int(len(filtered_frames["test"])),
            "output_dir": str(output_dir),
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

