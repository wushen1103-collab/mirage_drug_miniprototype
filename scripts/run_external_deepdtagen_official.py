from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import stats
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import data as DATA
from torch_geometric.loader import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import prepare_public_dti_official_splits
from mirage_mini.deepdtagen_adapter import prepare_deepdtagen_split_frame
from mirage_mini.deepdtagen_patch import patch_decoder_cross_attention_dims
from mirage_mini.external_baselines import compute_ranking_metrics, save_external_metrics, subsample_frame


DEEPDTAGEN_SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
DEEPDTAGEN_SEQ_DICT = {value: index + 1 for index, value in enumerate(DEEPDTAGEN_SEQ_VOC)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepdtagen-repo", required=True)
    parser.add_argument("--dataset", default="BindingDB_Kd")
    parser.add_argument("--split-method", default="cold_target", choices=["random", "cold_drug", "cold_target", "cold_pair"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    parser.add_argument("--inactive-threshold-nm", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-seq-len", type=int, default=1000)
    parser.add_argument("--max-target-tokens", type=int, default=128)
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


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == value for value in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == value for value in allowable_set] + [x not in allowable_set]


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na",
                "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb",
                "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge",
                "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg",
                "Pb", "Unknown",
            ],
        )
        + one_of_k_encoding(atom.GetDegree(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(11)))
        + one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
        + one_of_k_encoding_unk(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
        + [atom.GetIsAromatic()]
        + [atom.IsInRing()]
    )


def bond_features(bond):
    bond_type = bond.GetBondType()
    out = [0, 0, 0, 0, bond.GetBondTypeAsDouble()]
    if bond_type == Chem.rdchem.BondType.SINGLE:
        out = [1, 0, 0, 0, bond.GetBondTypeAsDouble()]
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        out = [0, 1, 0, 0, bond.GetBondTypeAsDouble()]
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        out = [0, 0, 1, 0, bond.GetBondTypeAsDouble()]
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        out = [0, 0, 0, 1, bond.GetBondTypeAsDouble()]
    return np.array(out)


def deepdtagen_smile_to_graph(smile: str):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smile}")
    atom_count = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / max(sum(feature), 1))

    edges = []
    for bond in mol.GetBonds():
        edge_feat = bond_features(bond)
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), {"edge_feats": edge_feat}))

    graph = nx.Graph()
    graph.add_edges_from(edges)
    graph = graph.to_directed()

    edge_index = []
    edge_feats = []
    for src, dst, feats in graph.edges(data=True):
        edge_index.append([src, dst])
        edge_feats.append(feats["edge_feats"])
    return atom_count, features, edge_index, edge_feats


def build_smile_graph(smiles: list[str]) -> dict[str, tuple]:
    return {smile: deepdtagen_smile_to_graph(smile) for smile in sorted(set(smiles))}


def deepdtagen_seq_cat(sequence: str, max_seq_len: int = 1000) -> np.ndarray:
    encoded = np.zeros(max_seq_len)
    for index, residue in enumerate(sequence[:max_seq_len]):
        encoded[index] = DEEPDTAGEN_SEQ_DICT.get(residue, 0)
    return encoded


def frame_to_deepdtagen_data_list(
    frame: pd.DataFrame,
    tokenizer,
    smile_graph: dict[str, tuple],
    *,
    max_seq_len: int,
    max_target_tokens: int,
) -> list[DATA.Data]:
    parsed_targets = [
        torch.LongTensor(tokenizer.parse(smiles)[:max_target_tokens])
        for smiles in frame["target_smiles"].astype(str)
    ]
    pad_token = tokenizer.s2i["<pad>"]
    padded_targets = pad_sequence(parsed_targets, batch_first=True, padding_value=pad_token)

    data_list = []
    for row_index, (_, row) in enumerate(frame.reset_index(drop=True).iterrows()):
        smiles = str(row["compound_iso_smiles"])
        target = deepdtagen_seq_cat(str(row["target_sequence"]), max_seq_len=max_seq_len)
        label = float(row["affinity"])
        c_size, features, edge_index, edge_feats = smile_graph[smiles]
        datum = DATA.Data(
            x=torch.tensor(np.asarray(features), dtype=torch.float32),
            edge_index=torch.tensor(np.asarray(edge_index), dtype=torch.long).t().contiguous(),
            edge_attr=torch.tensor(np.asarray(edge_feats), dtype=torch.float32),
            y=torch.tensor([label], dtype=torch.float32),
        )
        datum.target = torch.LongTensor([target])
        datum.target_seq = padded_targets[row_index].unsqueeze(0).long()
        datum.c_size = torch.LongTensor([c_size])
        data_list.append(datum)
    return data_list


def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            prediction, _, _, _ = model(batch)
            preds.append(prediction.view(-1).detach().cpu())
            labels.append(batch.y.view(-1).detach().cpu())
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def build_loaders(args: argparse.Namespace, tokenizer):
    splits = prepare_public_dti_official_splits(
        dataset_name=args.dataset,
        cache_dir=Path(args.cache_dir),
        split_method=args.split_method,
        seed=args.seed,
        activity_threshold_nm=args.activity_threshold_nm,
        inactive_threshold_nm=args.inactive_threshold_nm,
    )
    split_frames = {
        "train": prepare_deepdtagen_split_frame(
            subsample_frame(splits["train"], args.max_train_samples, args.seed),
            dataset_name=args.dataset,
        ),
        "val": prepare_deepdtagen_split_frame(
            subsample_frame(splits["val"], args.max_val_samples, args.seed + 1),
            dataset_name=args.dataset,
        ),
        "test": prepare_deepdtagen_split_frame(
            subsample_frame(splits["test"], args.max_test_samples, args.seed + 2),
            dataset_name=args.dataset,
        ),
    }

    all_smiles = pd.concat(
        [frame["compound_iso_smiles"] for frame in split_frames.values()],
        ignore_index=True,
    )
    smile_graph = build_smile_graph(all_smiles.astype(str).tolist())

    loaders = {}
    for split_name, frame in split_frames.items():
        data_list = frame_to_deepdtagen_data_list(
            frame,
            tokenizer,
            smile_graph,
            max_seq_len=args.max_seq_len,
            max_target_tokens=args.max_target_tokens,
        )
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

    if str(Path(args.deepdtagen_repo)) not in sys.path:
        sys.path.insert(0, str(Path(args.deepdtagen_repo)))
    from FetterGrad import FetterGrad
    from model import DeepDTAGen
    from utils import Tokenizer

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
    converted_splits = {
        "train": prepare_deepdtagen_split_frame(
            subsample_frame(raw_splits["train"], args.max_train_samples, args.seed),
            dataset_name=args.dataset,
        ),
        "val": prepare_deepdtagen_split_frame(
            subsample_frame(raw_splits["val"], args.max_val_samples, args.seed + 1),
            dataset_name=args.dataset,
        ),
        "test": prepare_deepdtagen_split_frame(
            subsample_frame(raw_splits["test"], args.max_test_samples, args.seed + 2),
            dataset_name=args.dataset,
        ),
    }
    for split_name, split_df in converted_splits.items():
        split_df.to_csv(output_dir / f"{split_name}_official_split.csv", index=False)

    all_target_smiles = pd.concat(
        [frame["target_smiles"] for frame in converted_splits.values()],
        ignore_index=True,
    )
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(all_target_smiles.astype(str).tolist()))

    split_frames, loaders = build_loaders(args, tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepDTAGen(tokenizer)
    patch_count = patch_decoder_cross_attention_dims(model, memory_dim=model.hidden_dim)
    if patch_count:
        print(json.dumps({"patched_decoder_layers": patch_count, "memory_dim": model.hidden_dim}), flush=True)
    model = model.to(device)
    optimizer = FetterGrad(torch.optim.Adam(model.parameters(), lr=args.learning_rate))
    mse_loss = nn.MSELoss()

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
            prediction, _, lm_loss, kl_loss = model(batch)
            prediction = prediction.view(-1)
            target = batch.y.view(-1)
            affinity_loss = mse_loss(prediction, target)
            total_loss = affinity_loss + lm_loss + (0.001 * kl_loss)

            optimizer.zero_grad()
            optimizer.ft_backward([total_loss, affinity_loss])
            optimizer.step()

            batch_size = int(target.shape[0])
            running_loss += float(total_loss.item()) * batch_size
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
        raise RuntimeError("DeepDTAGen training produced no checkpoint.")
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
        framework="deepdtagen",
        model="deepdtagen_multitask",
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

