from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mirage_mini.data import split_assay_cold, split_target_cold, split_temporal  # noqa: E402
from mirage_mini.external_baselines import (  # noqa: E402
    compute_ranking_metrics,
    prepare_external_regression_frame,
    save_external_metrics,
    subsample_frame,
)
from run_external_balm_official import _build_config_dict, _import_balm  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balm-repo", required=True)
    parser.add_argument("--benchmark-frame", required=True)
    parser.add_argument("--split-method", choices=["assay_cold", "target_cold", "temporal"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-variant", default="projection", choices=["baseline", "projection", "peft"])
    parser.add_argument("--protein-model", default="/home/test/wsk/hf_models/esm2_t30_150m_ur50d")
    parser.add_argument("--drug-model", default="/home/test/wsk/hf_models/chemberta_77m_mtr")
    parser.add_argument("--protein-max-length", type=int, default=1024)
    parser.add_argument("--drug-max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--projected-size", type=int, default=256)
    parser.add_argument("--projected-dropout", type=float, default=0.3)
    parser.add_argument("--warmup-steps-ratio", type=float, default=0.06)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_from_frame(frame: pd.DataFrame, split_method: str, seed: int) -> dict[str, pd.DataFrame]:
    if split_method == "assay_cold":
        return split_assay_cold(frame, seed=seed)
    if split_method == "target_cold":
        return split_target_cold(frame, seed=seed)
    if split_method == "temporal":
        return split_temporal(frame, seed=seed)
    raise ValueError(split_method)


def _prepare_split_frames(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(args.benchmark_frame)
    raw_splits = _split_from_frame(frame, split_method=args.split_method, seed=args.seed)
    limited = {
        "train": subsample_frame(raw_splits["train"], args.max_train_samples, args.seed),
        "val": subsample_frame(raw_splits["val"], args.max_val_samples, args.seed + 1),
        "test": subsample_frame(raw_splits["test"], args.max_test_samples, args.seed + 2),
    }
    return {
        split_name: prepare_external_regression_frame(split_df, dataset_name="CHEMBL_ASSAY")
        for split_name, split_df in limited.items()
    }


def _tokenize_unique_entities_truncated(dataset: pd.DataFrame, protein_tokenizer, drug_tokenizer, args):
    unique_proteins = dataset["Target"].astype(str).unique().tolist()
    unique_drugs = dataset["Drug"].astype(str).unique().tolist()
    protein_encoded = protein_tokenizer(
        unique_proteins,
        truncation=True,
        max_length=args.protein_max_length,
        add_special_tokens=True,
    )
    drug_encoded = drug_tokenizer(
        unique_drugs,
        truncation=True,
        max_length=args.drug_max_length,
        add_special_tokens=True,
    )
    protein_lookup = {
        protein: {"input_ids": ids, "attention_mask": mask}
        for protein, ids, mask in zip(
            unique_proteins,
            protein_encoded["input_ids"],
            protein_encoded["attention_mask"],
        )
    }
    drug_lookup = {
        drug: {"input_ids": ids, "attention_mask": mask}
        for drug, ids, mask in zip(
            unique_drugs,
            drug_encoded["input_ids"],
            drug_encoded["attention_mask"],
        )
    }
    return protein_lookup, drug_lookup


def _attach_strict_splits(
    trainer,
    split_frames: dict[str, pd.DataFrame],
    *,
    Dataset,
    DataCollatorWithPadding,
    tokenize_with_lookup,
    args: argparse.Namespace,
) -> None:
    trainer.pkd_lower_bound = float(split_frames["train"]["Y"].min())
    trainer.pkd_upper_bound = float(split_frames["train"]["Y"].max())

    for split_name, raw_frame in split_frames.items():
        work = raw_frame.reset_index(drop=True).copy()
        if trainer.model_configs.loss_function == "cosine_mse":
            pkd_range = trainer.pkd_upper_bound - trainer.pkd_lower_bound
            work["Y"] = 0.0 if pkd_range == 0 else (work["Y"] - trainer.pkd_lower_bound) / pkd_range * 2.0 - 1.0

        collator = DataCollatorWithPadding(
            protein_tokenizer=trainer.protein_tokenizer,
            drug_tokenizer=trainer.drug_tokenizer,
            padding="max_length",
            protein_max_length=args.protein_max_length,
            drug_max_length=args.drug_max_length,
            return_tensors="pt",
        )

        balm_frame = work[["sample_id", "drug_id", "target_id", "Drug", "Target", "Y"]].copy()
        protein_lookup, drug_lookup = _tokenize_unique_entities_truncated(
            balm_frame,
            trainer.protein_tokenizer,
            trainer.drug_tokenizer,
            args,
        )
        dataset = Dataset.from_pandas(balm_frame, preserve_index=False).map(
            lambda example: tokenize_with_lookup(example, protein_lookup, drug_lookup)
        )
        dataloader = torch.utils.data.DataLoader(
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


def _trainable_parameter_count(obj) -> int | None:
    model = getattr(obj, "model", None)
    if model is None:
        model = getattr(obj, "baseline_model", None)
    if model is None:
        return None
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT_NAME", "mirage_external_baselines")
    os.environ.setdefault("WANDB_ENTITY", "")

    (
        Configs,
        DataCollatorWithPadding,
        Dataset,
        Trainer,
        _pre_tokenize_unique_entities,
        tokenize_with_lookup,
    ) = _import_balm(Path(args.balm_repo).resolve())

    split_frames = _prepare_split_frames(args)
    for split_name, split_df in split_frames.items():
        split_df.to_csv(output_dir / f"{split_name}_strict_split.csv", index=False)

    config_args = argparse.Namespace(**vars(args))
    config_args.dataset = "CHEMBL_ASSAY"
    config_args.output_dir = str(output_dir)
    config_dict = _build_config_dict(config_args)
    configs = Configs(**config_dict)
    trainer = Trainer(
        configs=configs,
        wandb_entity=os.environ.get("WANDB_ENTITY", ""),
        wandb_project=os.environ.get("WANDB_PROJECT_NAME", "mirage_external_baselines"),
        outputs_dir=str(output_dir),
    )
    _attach_strict_splits(
        trainer,
        split_frames,
        Dataset=Dataset,
        DataCollatorWithPadding=DataCollatorWithPadding,
        tokenize_with_lookup=tokenize_with_lookup,
        args=args,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.setup_training()
    parameter_count = _trainable_parameter_count(trainer)
    start_train = time.perf_counter()
    trainer.train()
    train_seconds = time.perf_counter() - start_train
    start_infer = time.perf_counter()
    regression_metrics = trainer.test("test", save_prediction=True)
    inference_seconds = time.perf_counter() - start_infer

    pred_df = pd.read_csv(output_dir / "test_prediction.csv")
    if "Unnamed: 0" in pred_df.columns:
        pred_df = pred_df.drop(columns=["Unnamed: 0"])
    pred_df = pred_df.reset_index(drop=True)
    test_frame = split_frames["test"].reset_index(drop=True)
    if len(test_frame) != len(pred_df):
        raise ValueError(f"Prediction rows ({len(pred_df)}) do not match test rows ({len(test_frame)}).")

    merged = test_frame[
        ["sample_id", "drug_id", "target_id", "Drug", "Target", "binary_label", "label_raw_nm", "Y"]
    ].copy()
    merged["pkd_label"] = pred_df["label"]
    merged["pkd_prediction"] = pred_df["prediction"]
    merged.to_csv(output_dir / "test_prediction_with_binary.csv", index=False)

    metrics = {
        **compute_ranking_metrics(merged["binary_label"].to_numpy(), merged["pkd_prediction"].to_numpy()),
        "rmse": float(regression_metrics["test/rmse"]),
        "pearson": float(regression_metrics["test/pearson"]),
        "spearman": float(regression_metrics["test/spearman"]),
        "ci": float(regression_metrics["test/ci"]),
    }
    peak_gpu_memory_mb = None
    gpu_name = None
    if torch.cuda.is_available():
        peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
        gpu_name = torch.cuda.get_device_name()
    run_meta = {
        "strict_benchmark_frame": str(Path(args.benchmark_frame).resolve()),
        "train_size": int(len(split_frames["train"])),
        "val_size": int(len(split_frames["val"])),
        "test_size": int(len(split_frames["test"])),
        "metric_split_policy": "test_clean_after_token_truncation_no_row_filtering",
        "token_truncation": True,
        "protein_max_length": int(args.protein_max_length),
        "drug_max_length": int(args.drug_max_length),
        "protein_model": args.protein_model,
        "drug_model": args.drug_model,
        "parameter_count": parameter_count,
        "train_seconds": float(train_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_seconds_per_1000": float(inference_seconds / max(len(test_frame), 1) * 1000.0),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
    }
    payload = save_external_metrics(
        output_dir=output_dir,
        framework="balm",
        model=f"balm_{args.model_variant}_strict",
        dataset="CHEMBL_ASSAY",
        split_mode=args.split_method,
        seed=args.seed,
        metric_split="test_clean",
        metrics=metrics,
        run_meta=run_meta,
    )
    (output_dir / "efficiency.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

