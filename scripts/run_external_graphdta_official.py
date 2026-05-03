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
from torch_geometric.loader import DataLoader

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
from mirage_mini.graphdta_adapter import (
    build_smile_graph,
    frame_to_graphdta_data_list,
    resolve_graphdta_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphdta-repo", required=True)
    parser.add_argument("--dataset", default="BindingDB_Kd")
    parser.add_argument("--split-method", default="cold_target", choices=["random", "cold_drug", "cold_target", "cold_pair"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-variant", default="gin", choices=["gin", "gat", "gat_gcn", "gcn"])
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    parser.add_argument("--inactive-threshold-nm", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--graph-workers", type=int, default=1)
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


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch).view(-1)
            preds.append(output.detach().cpu())
            labels.append(batch.y.view(-1).detach().cpu())
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

    all_smiles = pd.concat([frame["Drug"] for frame in split_frames.values()], ignore_index=True)
    smile_graph = build_smile_graph(all_smiles.tolist(), num_workers=args.graph_workers)

    loaders: dict[str, DataLoader] = {}
    for split_name, split_frame in split_frames.items():
        data_list = frame_to_graphdta_data_list(split_frame, smile_graph=smile_graph)
        loaders[split_name] = DataLoader(
            data_list,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
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

    ModelClass = resolve_graphdta_model(args.graphdta_repo, args.model_variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelClass().to(device)
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
        for batch in loaders["train"]:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch).view(-1)
            loss = loss_fn(prediction, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            batch_size = int(batch.y.shape[0])
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        val_true, val_pred = evaluate(model, loaders["val"], device)
        val_rmse = rmse(val_true, val_pred)
        train_loss = running_loss / max(sample_count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_rmse": val_rmse})
        print(json.dumps(history[-1]), flush=True)

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
        raise RuntimeError("GraphDTA training produced no checkpoint.")
    model.load_state_dict(best_state)

    test_true, test_pred = evaluate(model, loaders["test"], device)
    test_frame = split_frames["test"].reset_index(drop=True).copy()
    if len(test_frame) != len(test_pred):
        raise ValueError(f"Prediction rows ({len(test_pred)}) do not match test rows ({len(test_frame)}).")

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
        framework="graphdta",
        model=f"graphdta_{args.model_variant}",
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
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

