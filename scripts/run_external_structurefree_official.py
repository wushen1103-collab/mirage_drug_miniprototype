from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import prepare_public_dti_official_splits
from mirage_mini.external_baselines import (
    compute_ranking_metrics,
    prepare_external_regression_frame,
    save_external_metrics,
    subsample_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structurefree-repo", required=True)
    parser.add_argument("--dataset", default="BindingDB_Kd")
    parser.add_argument("--split-method", default="cold_target", choices=["random", "cold_drug", "cold_target", "cold_pair"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protein-model", required=True)
    parser.add_argument("--drug-model", required=True)
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    parser.add_argument("--inactive-threshold-nm", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-molecule-length", type=int, default=128)
    parser.add_argument("--max-protein-length", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(stats.spearmanr(y_true, y_pred)[0])


def ci(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    order = np.argsort(y_true)
    y_true = y_true[order]
    y_pred = y_pred[order]
    i = len(y_true) - 1
    j = i - 1
    z = 0.0
    score = 0.0
    while i > 0:
        while j >= 0:
            if y_true[i] > y_true[j]:
                z += 1
                diff = y_pred[i] - y_pred[j]
                if diff > 0:
                    score += 1
                elif diff == 0:
                    score += 0.5
            j -= 1
        i -= 1
        j = i - 1
    return float(score / z) if z else float("nan")


class StructureFreeDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        return {
            "molecule": str(row["Drug"]),
            "protein": str(row["Target"]),
            "label": float(row["Y"]),
        }


class StructureFreeCollator:
    def __init__(
        self,
        *,
        molecule_tokenizer,
        protein_tokenizer,
        max_molecule_length: int,
        max_protein_length: int,
    ):
        self.molecule_tokenizer = molecule_tokenizer
        self.protein_tokenizer = protein_tokenizer
        self.max_molecule_length = max_molecule_length
        self.max_protein_length = max_protein_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        molecules = [item["molecule"] for item in batch]
        proteins = [item["protein"] for item in batch]
        labels = [item["label"] for item in batch]

        molecule_encodings = self.molecule_tokenizer(
            molecules,
            padding="max_length",
            truncation=True,
            max_length=self.max_molecule_length,
            return_tensors="pt",
        )
        protein_encodings = self.protein_tokenizer(
            proteins,
            padding="max_length",
            truncation=True,
            max_length=self.max_protein_length,
            return_tensors="pt",
        )
        return {
            "molecule_input_ids": molecule_encodings["input_ids"],
            "molecule_attention_mask": molecule_encodings["attention_mask"],
            "protein_input_ids": protein_encodings["input_ids"],
            "protein_attention_mask": protein_encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float32),
        }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    labels = batch["labels"].to(device)
    inputs = {
        key: value.to(device)
        for key, value in batch.items()
        if key != "labels"
    }
    return inputs, labels


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            inputs, batch_labels = move_batch_to_device(batch, device)
            outputs = model(inputs).view(-1)
            preds.append(outputs.detach().cpu())
            labels.append(batch_labels.detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def build_loaders(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], dict[str, DataLoader]]:
    splits = prepare_public_dti_official_splits(
        dataset_name=args.dataset,
        cache_dir=Path(args.cache_dir),
        split_method=args.split_method,
        seed=args.seed,
        activity_threshold_nm=args.activity_threshold_nm,
        inactive_threshold_nm=args.inactive_threshold_nm,
    )
    external_splits = {
        name: prepare_external_regression_frame(frame, dataset_name=args.dataset)
        for name, frame in splits.items()
    }
    split_frames = {
        "train": subsample_frame(external_splits["train"], args.max_train_samples, args.seed),
        "val": subsample_frame(external_splits["val"], args.max_val_samples, args.seed + 1),
        "test": subsample_frame(external_splits["test"], args.max_test_samples, args.seed + 2),
    }

    molecule_tokenizer = AutoTokenizer.from_pretrained(args.drug_model)
    protein_tokenizer = AutoTokenizer.from_pretrained(args.protein_model)
    collator = StructureFreeCollator(
        molecule_tokenizer=molecule_tokenizer,
        protein_tokenizer=protein_tokenizer,
        max_molecule_length=args.max_molecule_length,
        max_protein_length=args.max_protein_length,
    )

    loaders = {}
    for split_name, frame in split_frames.items():
        loaders[split_name] = DataLoader(
            StructureFreeDataset(frame),
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collator,
        )
    return split_frames, loaders


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    structurefree_src = Path(args.structurefree_repo) / "src"
    if str(structurefree_src) not in sys.path:
        sys.path.insert(0, str(structurefree_src))
    from models import AffinityPredictor, DrugTargetInteractionLoss

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frames, loaders = build_loaders(args)
    for split_name, split_df in split_frames.items():
        split_df.to_csv(output_dir / f"{split_name}_official_split.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AffinityPredictor(
        protein_model_name=args.protein_model,
        molecule_model_name=args.drug_model,
        hidden_sizes=[1024, 768, 512, 256, 1],
        inception_out_channels=256,
        dropout=0.05,
    ).to(device)
    loss_fn = DrugTargetInteractionLoss(alpha=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        sample_count = 0

        for step, batch in enumerate(loaders["train"], start=1):
            inputs, labels = move_batch_to_device(batch, device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=torch.bfloat16 if torch.cuda.is_available() else None):
                predictions = model(inputs).view(-1)
                loss = loss_fn(predictions, labels.view(-1))
                loss = loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()

            if step % args.gradient_accumulation_steps == 0 or step == len(loaders["train"]):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            batch_size = int(labels.shape[0])
            running_loss += float(loss.item()) * args.gradient_accumulation_steps * batch_size
            sample_count += batch_size

        val_true, val_pred = evaluate(model, loaders["val"], device)
        val_rmse = rmse(val_true, val_pred)
        train_loss = running_loss / max(sample_count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse})
        print(json.dumps(history[-1]), flush=True)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            patience_counter = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(best_state, output_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})", flush=True)
                break

    if best_state is None:
        raise RuntimeError("StructureFree-DTA training produced no checkpoint.")
    model.load_state_dict(best_state)

    test_true, test_pred = evaluate(model, loaders["test"], device)
    test_frame = split_frames["test"].reset_index(drop=True).copy()
    merged = test_frame[
        ["sample_id", "drug_id", "target_id", "Drug", "Target", "binary_label", "label_raw_nm", "Y"]
    ].copy()
    merged["pkd_label"] = test_true
    merged["pkd_prediction"] = test_pred
    merged.to_csv(output_dir / "test_prediction_with_binary.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    metrics = {
        **compute_ranking_metrics(merged["binary_label"].to_numpy(), merged["pkd_prediction"].to_numpy()),
        "rmse": rmse(test_true, test_pred),
        "pearson": pearson(test_true, test_pred),
        "spearman": spearman(test_true, test_pred),
        "ci": ci(test_true, test_pred),
    }
    payload = save_external_metrics(
        output_dir=output_dir,
        framework="structurefree_dta",
        model="structurefree_residual_inception",
        dataset=args.dataset,
        split_mode=args.split_method,
        seed=args.seed,
        metric_split="test_clean",
        metrics=metrics,
        run_meta={
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "train_size": int(len(split_frames["train"])),
            "val_size": int(len(split_frames["val"])),
            "test_size": int(len(split_frames["test"])),
            "protein_model": args.protein_model,
            "drug_model": args.drug_model,
            "output_dir": str(output_dir),
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

