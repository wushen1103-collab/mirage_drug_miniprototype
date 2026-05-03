from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import prepare_public_dti_official_splits
from mirage_mini.external_compat import (
    ensure_transformers_modeling_utils_compat,
    ensure_transformers_onnx_compat,
    ensure_transformers_pytorch_utils_compat,
)
from mirage_mini.external_baselines import compute_ranking_metrics, save_external_metrics, subsample_frame
from mirage_mini.mixingdta_adapter import build_mixingdta_entity_tables, frame_to_mixingdta_records, prepare_mixingdta_split_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixing-repo", required=True)
    parser.add_argument("--dataset", default="BindingDB_Kd")
    parser.add_argument("--split-method", default="cold_target", choices=["random", "cold_drug", "cold_target", "cold_pair"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    parser.add_argument("--inactive-threshold-nm", type=float, default=None)
    parser.add_argument("--drug-model", default="ibm/MoLFormer-XL-both-10pct")
    parser.add_argument("--drug-model-fallback", default=None)
    parser.add_argument("--protein-model", default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-batch-size-drug", type=int, default=16)
    parser.add_argument("--embed-batch-size-protein", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--dropout-mlp", type=float, default=0.15)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--max-protein-length", type=int, default=1022)
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


def _load_molformer_embeddings(
    drug_map: OrderedDict[str, str],
    *,
    model_name_or_path: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    ensure_transformers_onnx_compat()
    ensure_transformers_pytorch_utils_compat()
    ensure_transformers_modeling_utils_compat()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
    model.eval()

    items = list(drug_map.items())
    embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            tokenized = tokenizer([smiles for _, smiles in batch], return_tensors="pt", padding=True, truncation=True)
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            outputs = model(**tokenized)
            hidden = outputs.last_hidden_state.detach().cpu()
            mask = tokenized["attention_mask"].detach().cpu()
            for row_index, (drug_id, _) in enumerate(batch):
                valid_len = int(mask[row_index].sum().item())
                embeddings[str(drug_id)] = hidden[row_index, :valid_len].clone()
    return embeddings


def _embedding_dict_has_nan(embeddings: dict[str, torch.Tensor]) -> bool:
    return any(torch.isnan(tensor).any().item() for tensor in embeddings.values())


def _load_esm_embeddings(
    protein_map: OrderedDict[str, str],
    *,
    model_name_or_path: str,
    batch_size: int,
    device: torch.device,
    max_length: int,
) -> dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    items = list(protein_map.items())
    embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            tokenized = tokenizer(
                [sequence for _, sequence in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            tokenized = {key: value.to(device) for key, value in tokenized.items()}
            outputs = model(**tokenized)
            hidden = outputs.last_hidden_state.detach().cpu()
            mask = tokenized["attention_mask"].detach().cpu()
            for row_index, (target_id, _) in enumerate(batch):
                valid_len = int(mask[row_index].sum().item())
                start_idx = 1
                end_idx = max(start_idx + 1, valid_len - 1)
                embeddings[str(target_id)] = hidden[row_index, start_idx:end_idx].clone()
    return embeddings


def _pad_tensor_list(tensors: list[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    padded = pad_sequence(tensors, batch_first=True).to(device)
    masks = [torch.ones(tensor.size(0), dtype=torch.bool) for tensor in tensors]
    padded_mask = pad_sequence(masks, batch_first=True, padding_value=0).to(device)
    return padded, padded_mask


def _records_to_tensors(records, drug_to_index, protein_to_index):
    drug_idx = torch.tensor([drug_to_index[str(item[0])] for item in records], dtype=torch.long)
    protein_idx = torch.tensor([protein_to_index[str(item[2])] for item in records], dtype=torch.long)
    labels = torch.tensor([float(item[-1]) for item in records], dtype=torch.float32)
    return drug_idx, protein_idx, labels


def evaluate(model, loader, drug_embeddings, protein_embeddings, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for drug_idx, protein_idx, affinity in loader:
            drug_indices = drug_idx.view(-1).tolist()
            protein_indices = protein_idx.view(-1).tolist()
            batch_drug, mask_drug = _pad_tensor_list([drug_embeddings[index] for index in drug_indices], device=device)
            batch_protein, mask_protein = _pad_tensor_list([protein_embeddings[index] for index in protein_indices], device=device)
            prediction, _ = model(batch_drug, batch_protein, mask_drug, mask_protein)
            preds.append(prediction.view(-1).detach().cpu())
            labels.append(affinity.view(-1).detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    mixing_repo = Path(args.mixing_repo).resolve()
    meeta_root = mixing_repo / "MEETA"
    if str(meeta_root) not in sys.path:
        sys.path.insert(0, str(meeta_root))

    from model import DTA
    from utils import dict2namespace

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
        "train": prepare_mixingdta_split_frame(
            subsample_frame(raw_splits["train"], args.max_train_samples, args.seed),
            dataset_name=args.dataset,
        ),
        "val": prepare_mixingdta_split_frame(
            subsample_frame(raw_splits["val"], args.max_val_samples, args.seed + 1),
            dataset_name=args.dataset,
        ),
        "test": prepare_mixingdta_split_frame(
            subsample_frame(raw_splits["test"], args.max_test_samples, args.seed + 2),
            dataset_name=args.dataset,
        ),
    }
    for split_name, split_df in split_frames.items():
        split_df.to_csv(output_dir / f"{split_name}_official_split.csv", index=False)

    drugs, proteins = build_mixingdta_entity_tables(split_frames)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    drug_embedding_dict = _load_molformer_embeddings(
        drugs,
        model_name_or_path=args.drug_model,
        batch_size=args.embed_batch_size_drug,
        device=device,
    )
    drug_model_used = args.drug_model
    fallback_applied = False
    if _embedding_dict_has_nan(drug_embedding_dict):
        if not args.drug_model_fallback:
            raise RuntimeError(
                f"Drug encoder '{args.drug_model}' produced NaN embeddings. "
                "Provide --drug-model-fallback to retry with a stable encoder."
            )
        print(
            f"Drug encoder '{args.drug_model}' produced NaN embeddings; retrying with fallback "
            f"'{args.drug_model_fallback}'.",
            flush=True,
        )
        drug_embedding_dict = _load_molformer_embeddings(
            drugs,
            model_name_or_path=args.drug_model_fallback,
            batch_size=args.embed_batch_size_drug,
            device=device,
        )
        if _embedding_dict_has_nan(drug_embedding_dict):
            raise RuntimeError(
                f"Fallback drug encoder '{args.drug_model_fallback}' also produced NaN embeddings."
            )
        drug_model_used = args.drug_model_fallback
        fallback_applied = True
    protein_embedding_dict = _load_esm_embeddings(
        proteins,
        model_name_or_path=args.protein_model,
        batch_size=args.embed_batch_size_protein,
        device=device,
        max_length=args.max_protein_length,
    )
    torch.save(drug_embedding_dict, output_dir / "molformer_embeddings.pt")
    torch.save(protein_embedding_dict, output_dir / "protein_embeddings.pt")

    drug_ids = list(drugs.keys())
    protein_ids = list(proteins.keys())
    drug_to_index = {drug_id: index for index, drug_id in enumerate(drug_ids)}
    protein_to_index = {target_id: index for index, target_id in enumerate(protein_ids)}
    drug_embeddings_by_index = {drug_to_index[key]: value for key, value in drug_embedding_dict.items()}
    protein_embeddings_by_index = {protein_to_index[key]: value for key, value in protein_embedding_dict.items()}

    records = {split_name: frame_to_mixingdta_records(split_df) for split_name, split_df in split_frames.items()}
    train_loader = DataLoader(TensorDataset(*_records_to_tensors(records["train"], drug_to_index, protein_to_index)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(*_records_to_tensors(records["val"], drug_to_index, protein_to_index)), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(*_records_to_tensors(records["test"], drug_to_index, protein_to_index)), batch_size=args.batch_size, shuffle=False)

    if "esm2_t12_35M" in args.protein_model:
        protein_mode = "esm2_t12_35M_UR50D"
    elif "esm2_t6_8M" in args.protein_model:
        protein_mode = "esm2_t6_8M_UR50D"
    else:
        raise ValueError("Unsupported protein model for MixingDTA wrapper. Use esm2_t12_35M_UR50D or esm2_t6_8M_UR50D.")

    config = dict2namespace(
        {
            "n_embd": args.n_embd,
            "dropout_qkv": 0.0,
            "dropout_mlp": args.dropout_mlp,
            "masking": True,
            "binary_classification": False,
            "max_seq_len": max(512, args.max_protein_length + 8),
            "cross_pos_bias_switch": False,
            "drug": "MolFormer",
            "protein": protein_mode,
        }
    )
    model = DTA(config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_state = None
    best_epoch = -1
    best_val_rmse = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        sample_count = 0
        for drug_idx, protein_idx, affinity in train_loader:
            drug_indices = drug_idx.view(-1).tolist()
            protein_indices = protein_idx.view(-1).tolist()
            batch_drug, mask_drug = _pad_tensor_list([drug_embeddings_by_index[index] for index in drug_indices], device=device)
            batch_protein, mask_protein = _pad_tensor_list([protein_embeddings_by_index[index] for index in protein_indices], device=device)
            affinity = affinity.to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(batch_drug, batch_protein, mask_drug, mask_protein)
            loss = loss_fn(prediction.view(-1), affinity.view(-1))
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "MixingDTA loss became non-finite. Check upstream drug/protein embeddings."
                )
            loss.backward()
            optimizer.step()

            batch_size = int(affinity.shape[0])
            running_loss += float(loss.item()) * batch_size
            sample_count += batch_size

        val_true, val_pred = evaluate(model, val_loader, drug_embeddings_by_index, protein_embeddings_by_index, device)
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
        raise RuntimeError("MixingDTA training produced no checkpoint.")

    model.load_state_dict(best_state)
    test_true, test_pred = evaluate(model, test_loader, drug_embeddings_by_index, protein_embeddings_by_index, device)

    merged = split_frames["test"].copy()
    merged.rename(columns={"compound_iso_smiles": "Drug", "target_sequence": "Target"}, inplace=True)
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
        framework="mixingdta",
        model="mixingdta_meeta_nomixup",
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
            "drug_model": drug_model_used,
            "drug_model_requested": args.drug_model,
            "drug_model_fallback": args.drug_model_fallback,
            "drug_model_fallback_applied": fallback_applied,
            "protein_model": args.protein_model,
            "output_dir": str(output_dir),
        },
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

