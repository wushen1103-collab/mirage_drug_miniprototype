from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import prepare_benchmark_splits
from mirage_mini.deepdta_adapter import (
    DEEPDTA_MAX_PROTEIN_LEN,
    DEEPDTA_MAX_SMILES_LEN,
    DeepDTARegressor,
    DeepDTATensorDataset,
)
from mirage_mini.external_baselines import (
    compute_ranking_metrics,
    prepare_external_regression_frame,
    save_external_metrics,
    subsample_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BindingDB_Kd")
    parser.add_argument(
        "--split-method",
        default="cold_target",
        choices=["random", "cold_drug", "cold_target", "cold_pair", "target_cold", "assay_cold", "temporal"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    parser.add_argument("--inactive-threshold-nm", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--num-filters", type=int, default=32)
    parser.add_argument("--smiles-kernel-size", type=int, default=4)
    parser.add_argument("--protein-kernel-size", type=int, default=8)
    parser.add_argument("--max-smiles-len", type=int, default=DEEPDTA_MAX_SMILES_LEN)
    parser.add_argument("--max-protein-len", type=int, default=DEEPDTA_MAX_PROTEIN_LEN)
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


def collate_deepdta_batch(batch):
    smiles, proteins, labels = zip(*batch)
    return (
        torch.stack(smiles, dim=0),
        torch.stack(proteins, dim=0),
        torch.stack(labels, dim=0),
    )


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for smiles, proteins, batch_labels in loader:
            smiles = smiles.to(device)
            proteins = proteins.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(smiles, proteins)
            preds.append(outputs.detach().cpu())
            labels.append(batch_labels.detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def build_loaders(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], dict[str, DataLoader]]:
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
    split_frames = {
        "train": subsample_frame(external_splits["train"], args.max_train_samples, args.seed),
        "val": subsample_frame(external_splits["val"], args.max_val_samples, args.seed + 1),
        "test": subsample_frame(external_splits["test"], args.max_test_samples, args.seed + 2),
    }

    loaders: dict[str, DataLoader] = {}
    for split_name, frame in split_frames.items():
        loaders[split_name] = DataLoader(
            DeepDTATensorDataset(
                frame,
                max_smiles_len=args.max_smiles_len,
                max_protein_len=args.max_protein_len,
            ),
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_deepdta_batch,
        )
    return split_frames, loaders


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frames, loaders = build_loaders(args)
    for split_name, split_df in split_frames.items():
        split_df.to_csv(output_dir / f"{split_name}_official_split.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepDTARegressor(
        embedding_dim=args.embedding_dim,
        num_filters=args.num_filters,
        smiles_kernel_size=args.smiles_kernel_size,
        protein_kernel_size=args.protein_kernel_size,
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0

        for smiles, proteins, labels in loaders["train"]:
            smiles = smiles.to(device)
            proteins = proteins.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model(smiles, proteins)
            loss = loss_fn(predictions, labels.view(-1))
            loss.backward()
            optimizer.step()

            batch_size = int(labels.shape[0])
            running_loss += float(loss.item()) * batch_size
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
        raise RuntimeError("DeepDTA training produced no checkpoint.")
    model.load_state_dict(best_state)

    test_true, test_pred = evaluate(model, loaders["test"], device)
    test_frame = split_frames["test"].reset_index(drop=True).copy()
    merged = test_frame[
        ["sample_id", "drug_id", "target_id", "Drug", "Target", "binary_label", "label_raw_nm", "Y"]
    ].copy()
    merged["pkd_label"] = test_true
    merged["pkd_prediction"] = test_pred
    merged.to_csv(output_dir / "test_prediction_with_binary.csv", index=False)

    metrics = {
        **compute_ranking_metrics(merged["binary_label"].to_numpy(), merged["pkd_prediction"].to_numpy()),
        "rmse": rmse(test_true, test_pred),
        "pearson": pearson(test_true, test_pred),
        "spearman": spearman(test_true, test_pred),
        "ci": ci(test_true, test_pred),
    }
    payload = save_external_metrics(
        output_dir=output_dir,
        framework="deepdta",
        model="deepdta",
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
            "output_dir": str(output_dir),
            "embedding_dim": args.embedding_dim,
            "num_filters": args.num_filters,
            "smiles_kernel_size": args.smiles_kernel_size,
            "protein_kernel_size": args.protein_kernel_size,
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

