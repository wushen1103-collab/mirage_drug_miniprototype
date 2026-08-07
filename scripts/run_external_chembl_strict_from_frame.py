from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as GeometricDataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import split_assay_cold, split_target_cold, split_temporal  # noqa: E402
from mirage_mini.deepdta_adapter import (  # noqa: E402
    DEEPDTA_MAX_PROTEIN_LEN,
    DEEPDTA_MAX_SMILES_LEN,
    DeepDTARegressor,
    DeepDTATensorDataset,
)
from mirage_mini.external_baselines import (  # noqa: E402
    compute_ranking_metrics,
    prepare_external_regression_frame,
    save_external_metrics,
)
from mirage_mini.graphdta_adapter import (  # noqa: E402
    build_smile_graph,
    frame_to_graphdta_data_list,
    resolve_graphdta_model,
)
from mirage_mini.mgraphdta_adapter import (  # noqa: E402
    MGRAPHDTA_SEQ_DICT,
    build_mgraph_smile_graph,
    frame_to_mgraphdta_data_list,
    resolve_mgraphdta_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-frame", required=True)
    parser.add_argument("--model", choices=["deepdta", "graphdta", "mgraphdta"], required=True)
    parser.add_argument("--split-method", choices=["assay_cold", "target_cold", "temporal"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--graphdta-repo", default="/home/test/wsk/Finish/ModTune/experiments/docking_free/GraphDTA")
    parser.add_argument("--mgraphdta-repo", default="/tmp/mgraphdta_wrapper_revision_20260803")
    parser.add_argument("--graphdta-variant", default="gin", choices=["gin", "gat", "gat_gcn", "gcn"])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--graph-workers", type=int, default=1)
    parser.add_argument("--deepdta-embedding-dim", type=int, default=128)
    parser.add_argument("--deepdta-num-filters", type=int, default=32)
    parser.add_argument("--deepdta-smiles-kernel-size", type=int, default=4)
    parser.add_argument("--deepdta-protein-kernel-size", type=int, default=8)
    parser.add_argument("--mgraph-block-num", type=int, default=3)
    parser.add_argument("--mgraph-embedding-size", type=int, default=128)
    parser.add_argument("--mgraph-filter-num", type=int, default=32)
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


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    order = np.argsort(y_true)
    y_true = y_true[order]
    y_pred = y_pred[order]
    denom = 0.0
    score = 0.0
    for i in range(1, len(y_true)):
        valid = y_true[i] > y_true[:i]
        if not np.any(valid):
            continue
        diff = y_pred[i] - y_pred[:i][valid]
        denom += float(len(diff))
        score += float(np.sum(diff > 0) + 0.5 * np.sum(diff == 0))
    return float(score / denom) if denom else float("nan")


def split_from_frame(frame: pd.DataFrame, split_method: str, seed: int) -> dict[str, pd.DataFrame]:
    if split_method == "assay_cold":
        return split_assay_cold(frame, seed=seed)
    if split_method == "target_cold":
        return split_target_cold(frame, seed=seed)
    if split_method == "temporal":
        return split_temporal(frame, seed=seed)
    raise ValueError(split_method)


def prepare_splits(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(args.benchmark_frame)
    raw_splits = split_from_frame(frame, args.split_method, args.seed)
    return {
        name: prepare_external_regression_frame(split_frame, dataset_name="CHEMBL_ASSAY")
        for name, split_frame in raw_splits.items()
    }


def deepdta_collate(batch):
    smiles, proteins, labels = zip(*batch)
    return torch.stack(smiles), torch.stack(proteins), torch.stack(labels)


def build_loaders(args: argparse.Namespace, splits: dict[str, pd.DataFrame]):
    if args.model == "deepdta":
        return {
            name: TorchDataLoader(
                DeepDTATensorDataset(
                    split,
                    max_smiles_len=DEEPDTA_MAX_SMILES_LEN,
                    max_protein_len=DEEPDTA_MAX_PROTEIN_LEN,
                ),
                batch_size=args.batch_size,
                shuffle=(name == "train"),
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
                collate_fn=deepdta_collate,
            )
            for name, split in splits.items()
        }
    all_smiles = pd.concat([split["Drug"] for split in splits.values()], ignore_index=True)
    if args.model == "graphdta":
        smile_graph = build_smile_graph(all_smiles.tolist(), num_workers=args.graph_workers)
        return {
            name: GeometricDataLoader(
                frame_to_graphdta_data_list(split, smile_graph=smile_graph),
                batch_size=args.batch_size,
                shuffle=(name == "train"),
                num_workers=args.num_workers,
                pin_memory=torch.cuda.is_available(),
            )
            for name, split in splits.items()
        }
    smile_graph = build_mgraph_smile_graph(all_smiles.tolist(), num_workers=args.graph_workers)
    return {
        name: GeometricDataLoader(
            frame_to_mgraphdta_data_list(split, smile_graph=smile_graph),
            batch_size=args.batch_size,
            shuffle=(name == "train"),
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        for name, split in splits.items()
    }


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.model == "deepdta":
        return DeepDTARegressor(
            embedding_dim=args.deepdta_embedding_dim,
            num_filters=args.deepdta_num_filters,
            smiles_kernel_size=args.deepdta_smiles_kernel_size,
            protein_kernel_size=args.deepdta_protein_kernel_size,
        )
    if args.model == "graphdta":
        model_class = resolve_graphdta_model(args.graphdta_repo, args.graphdta_variant)
        return model_class()
    model_class = resolve_mgraphdta_model(args.mgraphdta_repo)
    return model_class(
        args.mgraph_block_num,
        len(MGRAPHDTA_SEQ_DICT) + 1,
        embedding_size=args.mgraph_embedding_size,
        filter_num=args.mgraph_filter_num,
        out_dim=1,
    )


def forward_batch(model: nn.Module, batch, device: torch.device, model_name: str):
    if model_name == "deepdta":
        smiles, proteins, labels = batch
        smiles = smiles.to(device)
        proteins = proteins.to(device)
        labels = labels.to(device)
        return model(smiles, proteins).view(-1), labels.view(-1)
    batch = batch.to(device)
    return model(batch).view(-1), batch.y.view(-1)


def evaluate(model: nn.Module, loader, device: torch.device, model_name: str) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    preds = []
    labels = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            pred, label = forward_batch(model, batch, device, model_name)
            preds.append(pred.detach().cpu())
            labels.append(label.detach().cpu())
    elapsed = time.perf_counter() - start
    return torch.cat(labels).numpy(), torch.cat(preds).numpy(), elapsed


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = prepare_splits(args)
    for split_name, split in splits.items():
        split.to_csv(output_dir / f"{split_name}_strict_split.csv", index=False)
    loaders = build_loaders(args, splits)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args).to(device)
    parameter_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    start_train = time.perf_counter()
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            pred, label = forward_batch(model, batch, device, args.model)
            loss = loss_fn(pred, label)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * int(label.shape[0])
            count += int(label.shape[0])
        val_true, val_pred, _ = evaluate(model, loaders["val"], device, args.model)
        val_rmse = rmse(val_true, val_pred)
        history.append({"epoch": epoch, "train_loss": running / max(count, 1), "val_rmse": val_rmse})
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
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
                break
    train_seconds = time.perf_counter() - start_train
    if best_state is None:
        raise RuntimeError("No checkpoint was produced.")
    model.load_state_dict(best_state)

    test_true, test_pred, inference_seconds = evaluate(model, loaders["test"], device, args.model)
    test_frame = splits["test"].reset_index(drop=True).copy()
    merged = test_frame[["sample_id", "drug_id", "target_id", "Drug", "Target", "binary_label", "label_raw_nm", "Y"]].copy()
    merged["pkd_label"] = test_true
    merged["pkd_prediction"] = test_pred
    merged.to_csv(output_dir / "test_prediction_with_binary.csv", index=False)

    peak_memory_mb = None
    gpu_name = None
    if torch.cuda.is_available():
        peak_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        gpu_name = torch.cuda.get_device_name(device)
    metrics = {
        **compute_ranking_metrics(merged["binary_label"].to_numpy(), merged["pkd_prediction"].to_numpy()),
        "rmse": rmse(test_true, test_pred),
        "mae": float(np.mean(np.abs(test_true - test_pred))),
        "pearson": pearson(test_true, test_pred),
        "spearman": spearman(test_true, test_pred),
        "ci": concordance_index(test_true, test_pred),
    }
    run_meta = {
        "strict_benchmark_frame": str(Path(args.benchmark_frame).resolve()),
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "train_size": int(len(splits["train"])),
        "val_size": int(len(splits["val"])),
        "test_size": int(len(splits["test"])),
        "parameter_count": parameter_count,
        "train_seconds": float(train_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_seconds_per_1000": float(inference_seconds / max(len(splits["test"]), 1) * 1000.0),
        "peak_gpu_memory_mb": peak_memory_mb,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    payload = save_external_metrics(
        output_dir=output_dir,
        framework=args.model,
        model=f"{args.model}_{args.graphdta_variant}" if args.model == "graphdta" else args.model,
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
