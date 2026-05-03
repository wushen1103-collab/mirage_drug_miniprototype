from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn

try:
    import resource
except ImportError:  # pragma: no cover - unavailable on Windows
    resource = None

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.caster_adapter import build_caster_matrix_inputs, prepare_caster_split_frame
from mirage_mini.data import prepare_public_dti_official_splits
from mirage_mini.external_compat import install_caster_download_patch
from mirage_mini.external_baselines import compute_ranking_metrics, save_external_metrics, subsample_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--caster-repo", required=True)
    parser.add_argument("--dataset", default="BindingDB_Kd")
    parser.add_argument("--split-method", default="cold_target", choices=["random", "cold_drug", "cold_target", "cold_pair"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--caster-data-root", default=None)
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    parser.add_argument("--inactive-threshold-nm", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batch-elements", type=int, default=2000000)
    parser.add_argument("--allow-complexed-pdb", action="store_true")
    parser.add_argument("--skip-pdb-download", action="store_true")
    parser.add_argument("--overwrite-pdb", action="store_true")
    parser.add_argument("--create-comp", action="store_true")
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


def get_open_file_soft_limit() -> int | None:
    if resource is None:
        return None
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit < 0:
        return None
    return int(soft_limit)


def resolve_caster_pool_workers(
    dataloader_workers: int,
    cpu_count: int | None = None,
    fd_soft_limit: int | None = None,
) -> int:
    requested_workers = max(1, int(dataloader_workers))
    available_cpus = max(1, int(cpu_count or (os.cpu_count() or 1)))
    resolved_workers = min(requested_workers, available_cpus)

    if fd_soft_limit is not None and fd_soft_limit > 0:
        # Leave headroom for shared-memory handles, PDB files, and DataLoader workers.
        fd_capped_workers = max(1, min(available_cpus, (int(fd_soft_limit) - 128) // 32))
        resolved_workers = min(resolved_workers, fd_capped_workers)

    return max(1, resolved_workers)


def _build_model(dataset, joint_gnn_cls):
    protein_gnn_kwargs = dict(
        base_conv="lbamodel",
        in_channels=dataset.metadata_dict["protein_node_features"],
        edge_dim=dataset.metadata_dict["protein_edge_features"],
        num_ntypes=dataset.metadata_dict["protein_node_types"],
        num_etypes=dataset.metadata_dict["protein_edge_types"],
        ntype_emb_dim=None,
        etype_emb_dim=None,
        num_convs=2,
        hidden_channels=(16, 4),
        edge_hidden_channels=(32, 1),
        out_channels=64,
        dropout_rate=0.2,
        activation="leaky_relu",
        aggr="sum",
    )
    molecule_gnn_kwargs = dict(
        base_conv="gine",
        in_channels=dataset.metadata_dict["molecule_node_features"],
        edge_dim=dataset.metadata_dict["molecule_edge_features"],
        num_ntypes=dataset.metadata_dict["molecule_node_types"],
        num_etypes=dataset.metadata_dict["molecule_edge_types"],
        ntype_emb_dim=None,
        etype_emb_dim=None,
        num_convs=2,
        hidden_channels=16,
        out_channels=64,
        dropout_rate=0.2,
        activation="leaky_relu",
        aggr="sum",
        gin_trainable_eps=True,
    )
    joint_gnn_kwargs = dict(
        residue_lin_depth=1,
        atom_lin_depth=1,
        n_attention_heads=8,
        attention_dropout=0.0,
        protein_lin_depth=1,
        molecule_lin_depth=1,
        pairwise_embedding_dim=512,
        out_lin_depth=1,
        out_lin_factor=0.5,
        out_lin_norm_type=None,
        activation="leaky_relu",
        dropout=0.1,
        element_pooling="mean",
        include_residual_stream=True,
        residual_dim_ff_scale=2,
        num_cross_attn_layers=1,
        include_post_pool_layernorm=False,
    )
    model = joint_gnn_cls(protein_gnn_kwargs, molecule_gnn_kwargs, **joint_gnn_kwargs)
    model._mirage_kwargs = {
        "protein_gnn_kwargs": protein_gnn_kwargs,
        "molecule_gnn_kwargs": molecule_gnn_kwargs,
        "joint_gnn_kwargs": joint_gnn_kwargs,
    }
    return model


def evaluate(model, loader, dataset, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for protein_graph, molecule_graph, affinity in loader:
            protein_graph = protein_graph.to(device)
            molecule_graph = molecule_graph.to(device)
            prediction, _ = model.forward_with_graphs(protein_graph, molecule_graph)
            preds.append(prediction.view(-1).detach().cpu())
            labels.append(affinity.view(-1).detach().cpu())
    pred_scaled = torch.cat(preds)
    label_scaled = torch.cat(labels)
    pred = dataset.unscale_target(pred_scaled).numpy()
    label = dataset.unscale_target(label_scaled).numpy()
    return label, pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    caster_repo = Path(args.caster_repo).resolve()
    if str(caster_repo) not in sys.path:
        sys.path.insert(0, str(caster_repo))

    import dataset.dual_dataset as dual_dataset
    from dataset.dual_dataset import PMD_DataLoader, ProteinMoleculeDataset
    from dataset import process_data as caster_process_data
    from models.joint_gnn import JointGNN
    install_caster_download_patch(caster_process_data)

    graph_build_workers = resolve_caster_pool_workers(
        dataloader_workers=args.num_workers,
        fd_soft_limit=get_open_file_soft_limit(),
    )
    dual_dataset.N_PROC = graph_build_workers
    print(
        json.dumps(
            {
                "graph_build_workers": graph_build_workers,
                "dataloader_workers": args.num_workers,
                "open_file_soft_limit": get_open_file_soft_limit(),
            }
        ),
        flush=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_splits = prepare_public_dti_official_splits(
        dataset_name=args.dataset,
        cache_dir=Path(args.cache_dir),
        split_method=args.split_method,
        seed=args.seed,
        activity_threshold_nm=args.activity_threshold_nm,
        inactive_threshold_nm=args.inactive_threshold_nm,
    )
    split_frames = {
        "train": prepare_caster_split_frame(
            subsample_frame(raw_splits["train"], args.max_train_samples, args.seed),
            dataset_name=args.dataset,
        ),
        "val": prepare_caster_split_frame(
            subsample_frame(raw_splits["val"], args.max_val_samples, args.seed + 1),
            dataset_name=args.dataset,
        ),
        "test": prepare_caster_split_frame(
            subsample_frame(raw_splits["test"], args.max_test_samples, args.seed + 2),
            dataset_name=args.dataset,
        ),
    }
    for split_name, split_df in split_frames.items():
        split_df.to_csv(output_dir / f"{split_name}_official_split.csv", index=False)

    merged_input = []
    for split_name, split_df in split_frames.items():
        tagged = split_df.copy()
        tagged["split"] = split_name
        merged_input.append(tagged)
    merged_input = pd.concat(merged_input, ignore_index=True)

    proteins, ligands, affinity = build_caster_matrix_inputs(split_frames)
    caster_data_root = Path(args.caster_data_root) if args.caster_data_root else (output_dir / "caster_dataset")
    caster_data_root.mkdir(parents=True, exist_ok=True)
    processed = caster_process_data.process_data(
        proteins=proteins,
        ligands=ligands,
        affinity=affinity,
        data_path=str(caster_data_root),
        pdb_dir_name="pdb_files",
        overwrite_csv=True,
        skip_pdb_dl=args.skip_pdb_download,
        overwrite_pdb=args.overwrite_pdb,
        allow_complexed_pdb=args.allow_complexed_pdb,
        create_comp=args.create_comp,
        reverse_comp_fold_order=False,
        verbose_pdb_dl=False,
        verbose_comp_fold=False,
    )
    processed = processed.merge(
        merged_input,
        on=["protein_id", "protein_sequence", "molecule_id", "molecule_smiles", "affinity_score"],
        how="inner",
    )
    if processed.empty:
        raise RuntimeError("CASTER-DTA preprocessing produced no valid pairs after structure filtering.")
    processed.to_csv(output_dir / "processed_pairs_with_splits.csv", index=False)

    dataset_kwargs = {
        "sparse_edges": False,
        "protein_dist_units": "angstroms",
        "protein_edge_thresh": 4,
        "protein_thresh_type": "dist",
        "protein_keep_selfloops": True,
        "protein_vector_features": True,
        "protein_include_esm2": False,
        "protein_include_residue_posenc": False,
        "protein_include_aa_props": True,
        "molecule_full_atomtype": False,
        "molecule_onehot_ordinal_feats": False,
        "molecule_include_selfloops": True,
        "scale_output": ["standardize"],
    }
    dataset = ProteinMoleculeDataset(processed, **dataset_kwargs)
    (output_dir / "dataset_kwargs.json").write_text(json.dumps(dataset_kwargs, indent=2), encoding="utf-8")

    split_indices = {
        split_name: processed.index[processed["split"].eq(split_name)].tolist()
        for split_name in ["train", "val", "test"]
    }
    train_ds = torch.utils.data.Subset(dataset, split_indices["train"])
    val_ds = torch.utils.data.Subset(dataset, split_indices["val"])
    test_ds = torch.utils.data.Subset(dataset, split_indices["test"])

    loader_kwargs = {
        "max_num": args.max_batch_elements,
        "max_batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": True,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": bool(args.num_workers),
    }
    train_loader = PMD_DataLoader(train_ds, **loader_kwargs)
    val_loader = PMD_DataLoader(val_ds, **{**loader_kwargs, "shuffle": False})
    test_loader = PMD_DataLoader(test_ds, **{**loader_kwargs, "shuffle": False})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(dataset, JointGNN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=2)
    loss_fn = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_state = None
    best_epoch = -1
    best_val_rmse = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for protein_graph, molecule_graph, affinity_values in train_loader:
            protein_graph = protein_graph.to(device)
            molecule_graph = molecule_graph.to(device)
            affinity_values = affinity_values.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                prediction, _ = model.forward_with_graphs(protein_graph, molecule_graph)
                loss = loss_fn(prediction.view(-1), affinity_values.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = int(affinity_values.shape[0])
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        val_true, val_pred = evaluate(model, val_loader, dataset, device)
        val_rmse = rmse(val_true, val_pred)
        scheduler.step(val_rmse)
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
        raise RuntimeError("CASTER-DTA training produced no checkpoint.")

    model.load_state_dict(best_state)
    test_true, test_pred = evaluate(model, test_loader, dataset, device)
    test_meta = processed.iloc[split_indices["test"]].reset_index(drop=True).copy()
    merged = test_meta[
        ["sample_id", "drug_id", "target_id", "molecule_smiles", "protein_sequence", "binary_label", "label_raw_nm"]
    ].copy()
    merged.rename(columns={"molecule_smiles": "Drug", "protein_sequence": "Target"}, inplace=True)
    merged["Y"] = test_true
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
        framework="caster_dta",
        model="caster_dta_joint_gnn",
        dataset=args.dataset,
        split_mode=args.split_method,
        seed=args.seed,
        metric_split="test_clean",
        metrics=metrics,
        run_meta={
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "train_size": int(len(train_ds)),
            "val_size": int(len(val_ds)),
            "test_size": int(len(test_ds)),
            "graph_build_workers": int(graph_build_workers),
            "dataloader_workers": int(args.num_workers),
            "structure_pairs_retained": int(len(processed)),
            "unique_proteins_retained": int(processed["protein_id"].nunique()),
            "unique_molecules_retained": int(processed["molecule_id"].nunique()),
            "output_dir": str(output_dir),
            "caster_data_root": str(caster_data_root),
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

