from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import prepare_public_dti_official_splits
from mirage_mini.external_baselines import compute_ranking_metrics, save_external_metrics, subsample_frame
from mirage_mini.kanpm_adapter import build_kanpm_entity_tables, prepare_kanpm_split_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kanpm-repo", required=True)
    parser.add_argument("--dataset", default="BindingDB_Kd")
    parser.add_argument("--split-method", default="cold_target", choices=["random", "cold_drug", "cold_target", "cold_pair"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protein-model", required=True)
    parser.add_argument("--drug-model", required=True)
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    parser.add_argument("--inactive-threshold-nm", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--drug-max-length", type=int, default=220)
    parser.add_argument("--protein-token-limit", type=int, default=1022)
    parser.add_argument("--protein-max-length", type=int, default=1024)
    parser.add_argument("--feature-batch-size-drug", type=int, default=32)
    parser.add_argument("--feature-batch-size-protein", type=int, default=1)
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


def build_feature_cache_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    feature_dir = output_dir / "feature_cache"
    feature_dir.mkdir(parents=True, exist_ok=True)
    return (
        feature_dir / "drug_features.pkl",
        feature_dir / "protein_features.pkl",
        feature_dir / "contact_maps.pkl",
    )


def _valid_token_span(attention_mask_row: torch.Tensor) -> slice:
    valid_len = int(attention_mask_row.sum().item())
    start = 1
    stop = max(start, valid_len - 1)
    return slice(start, stop)


def generate_drug_features(
    drug_table: pd.DataFrame,
    *,
    model_name_or_path: str,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    features = {"vec_dict": {}, "mat_dict": {}, "length_dict": {}}
    rows = drug_table[["drug_id", "drug_seq"]].to_dict("records")
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        smiles = [row["drug_seq"] for row in chunk]
        encodings = tokenizer(
            smiles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encodings = {key: value.to(device) for key, value in encodings.items()}
        with torch.no_grad():
            outputs = model(**encodings).last_hidden_state.detach().cpu()
        attention_mask = encodings["attention_mask"].detach().cpu()

        for index, row in enumerate(chunk):
            span = _valid_token_span(attention_mask[index])
            token_matrix = outputs[index, span, :].contiguous()
            features["mat_dict"][str(row["drug_id"])] = token_matrix
            features["vec_dict"][str(row["drug_id"])] = token_matrix.mean(dim=0)
            features["length_dict"][str(row["drug_id"])] = int(token_matrix.shape[0])

    return features


def generate_protein_features_and_contacts(
    protein_table: pd.DataFrame,
    *,
    model_name_or_path: str,
    token_limit: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict, dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, attn_implementation="eager").to(device)
    model.eval()

    features = {"vec_dict": {}, "mat_dict": {}, "length_dict": {}}
    contact_maps = {"contact_map": {}, "length_dict": {}}
    rows = protein_table[["prot_id", "prot_seq"]].to_dict("records")

    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        sequences = [row["prot_seq"][:token_limit] for row in chunk]
        encodings = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=token_limit,
            return_tensors="pt",
        )
        encodings = {key: value.to(device) for key, value in encodings.items()}
        with torch.no_grad():
            hidden = model(**encodings).last_hidden_state.detach().cpu()
            contacts = model.predict_contacts(
                encodings["input_ids"],
                encodings["attention_mask"],
            ).detach().cpu()
        attention_mask = encodings["attention_mask"].detach().cpu()

        for index, row in enumerate(chunk):
            valid_len = int(attention_mask[index].sum().item())
            token_matrix = hidden[index, :valid_len, :].contiguous()
            aa_length = max(valid_len - 2, 0)
            features["mat_dict"][str(row["prot_id"])] = token_matrix.numpy()
            features["vec_dict"][str(row["prot_id"])] = token_matrix.mean(dim=0).numpy()
            features["length_dict"][str(row["prot_id"])] = aa_length

            contact_map = contacts[index].numpy()
            contact_maps["contact_map"][str(row["prot_id"])] = contact_map
            contact_maps["length_dict"][str(row["prot_id"])] = int(contact_map.shape[0])

    return features, contact_maps


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch_data in loader:
            mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, drug_graph, protein_graph, affinity = batch_data
            prediction = model(
                mol_vec.to(device),
                mol_mat.to(device),
                mol_mat_mask.to(device),
                prot_vec.to(device),
                prot_mat.to(device),
                prot_mat_mask.to(device),
                drug_graph.to(device),
                protein_graph.to(device),
            ).view(-1)
            preds.append(prediction.detach().cpu())
            labels.append(affinity.view(-1).detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def build_loaders(args: argparse.Namespace):
    splits = prepare_public_dti_official_splits(
        dataset_name=args.dataset,
        cache_dir=Path(args.cache_dir),
        split_method=args.split_method,
        seed=args.seed,
        activity_threshold_nm=args.activity_threshold_nm,
        inactive_threshold_nm=args.inactive_threshold_nm,
    )
    split_frames = {
        "train": prepare_kanpm_split_frame(
            subsample_frame(splits["train"], args.max_train_samples, args.seed),
            dataset_name=args.dataset,
        ),
        "val": prepare_kanpm_split_frame(
            subsample_frame(splits["val"], args.max_val_samples, args.seed + 1),
            dataset_name=args.dataset,
        ),
        "test": prepare_kanpm_split_frame(
            subsample_frame(splits["test"], args.max_test_samples, args.seed + 2),
            dataset_name=args.dataset,
        ),
    }
    drug_table, protein_table = build_kanpm_entity_tables(split_frames)
    return split_frames, drug_table, protein_table


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    kanpm_code = Path(args.kanpm_repo) / "code"
    if str(kanpm_code) not in sys.path:
        sys.path.insert(0, str(kanpm_code))
    from model import MODEL as KanpmModel
    from model import ProteinGraphNet
    from MyDataset import CustomDataSet, my_collate_fn

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_frames, drug_table, protein_table = build_loaders(args)
    for split_name, split_df in split_frames.items():
        split_df.to_csv(output_dir / f"{split_name}_official_split.csv", index=False)
    drug_table.to_csv(output_dir / "kanpm_drugs.csv", index=False)
    protein_table.to_csv(output_dir / "kanpm_proteins.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    drug_cache_path, protein_cache_path, contact_cache_path = build_feature_cache_paths(output_dir)

    if drug_cache_path.exists():
        with drug_cache_path.open("rb") as handle:
            drug_features = pickle.load(handle)
    else:
        drug_features = generate_drug_features(
            drug_table,
            model_name_or_path=args.drug_model,
            max_length=args.drug_max_length,
            batch_size=args.feature_batch_size_drug,
            device=device,
        )
        with drug_cache_path.open("wb") as handle:
            pickle.dump(drug_features, handle)

    if protein_cache_path.exists() and contact_cache_path.exists():
        with protein_cache_path.open("rb") as handle:
            protein_features = pickle.load(handle)
        with contact_cache_path.open("rb") as handle:
            contact_maps = pickle.load(handle)
    else:
        protein_features, contact_maps = generate_protein_features_and_contacts(
            protein_table,
            model_name_or_path=args.protein_model,
            token_limit=args.protein_token_limit,
            batch_size=args.feature_batch_size_protein,
            device=device,
        )
        with protein_cache_path.open("wb") as handle:
            pickle.dump(protein_features, handle)
        with contact_cache_path.open("wb") as handle:
            pickle.dump(contact_maps, handle)

    protein_hidden_size = int(next(iter(protein_features["vec_dict"].values())).shape[0])
    drug_hidden_size = int(next(iter(drug_features["vec_dict"].values())).shape[0])
    hp = SimpleNamespace(
        drug_max_len=args.drug_max_length,
        prot_max_len=args.protein_max_length,
        mol2vec_dim=drug_hidden_size,
        protvec_dim=protein_hidden_size,
        dropout=0.2,
    )

    split_datasets = {name: CustomDataSet(frame, hp) for name, frame in split_frames.items()}
    loaders = {}
    for split_name, dataset in split_datasets.items():
        loader_workers = args.num_workers if split_name == "train" else 0
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split_name == "train"),
            num_workers=loader_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=loader_workers > 0,
            collate_fn=lambda batch, hp=hp: my_collate_fn(
                batch,
                device,
                hp,
                drug_table,
                protein_table,
                drug_features,
                protein_features,
                contact_maps,
            ),
        )

    model = KanpmModel(hp, device)
    model.protein_graph_model = ProteinGraphNet(
        n_output=128,
        num_features_xd=protein_hidden_size,
        output_dim=128,
        dropout=0.2,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_rmse = float("inf")
    best_epoch = -1
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for batch_data in loaders["train"]:
            mol_vec, prot_vec, mol_mat, mol_mat_mask, prot_mat, prot_mat_mask, drug_graph, protein_graph, affinity = batch_data
            mol_vec = mol_vec.to(device)
            prot_vec = prot_vec.to(device)
            mol_mat = mol_mat.to(device)
            mol_mat_mask = mol_mat_mask.to(device)
            prot_mat = prot_mat.to(device)
            prot_mat_mask = prot_mat_mask.to(device)
            drug_graph = drug_graph.to(device)
            protein_graph = protein_graph.to(device)
            affinity = affinity.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                mol_vec,
                mol_mat,
                mol_mat_mask,
                prot_vec,
                prot_mat,
                prot_mat_mask,
                drug_graph,
                protein_graph,
            ).view(-1)
            loss = loss_fn(prediction, affinity.view(-1))
            loss.backward()
            optimizer.step()

            batch_size = int(affinity.shape[0])
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
        raise RuntimeError("KANPM-DTA training produced no checkpoint.")
    model.load_state_dict(best_state)

    test_true, test_pred = evaluate(model, loaders["test"], device)
    test_frame = split_frames["test"].reset_index(drop=True).copy()
    merged = test_frame[
        ["sample_id", "drug_id", "target_id", "compound_iso_smiles", "target_sequence", "binary_label", "label_raw_nm", "affinity"]
    ].copy()
    merged.rename(
        columns={
            "compound_iso_smiles": "Drug",
            "target_sequence": "Target",
            "affinity": "Y",
        },
        inplace=True,
    )
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
        framework="kanpm_dta",
        model="kanpm_transformer_backbone",
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
            "drug_hidden_size": drug_hidden_size,
            "protein_hidden_size": protein_hidden_size,
            "output_dir": str(output_dir),
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

