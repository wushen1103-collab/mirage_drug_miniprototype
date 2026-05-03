from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data


GRAPHDTA_SEQ_VOC = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
GRAPHDTA_SEQ_DICT = {token: index + 1 for index, token in enumerate(GRAPHDTA_SEQ_VOC)}
GRAPHDTA_MAX_SEQ_LEN = 1000

GRAPHDTA_MODEL_SPECS = {
    "gin": ("models.ginconv", "GINConvNet"),
    "gat": ("models.gat", "GATNet"),
    "gat_gcn": ("models.gat_gcn", "GAT_GCN"),
    "gcn": ("models.gcn", "GCNNet"),
}


def graphdta_seq_cat(sequence: str, max_seq_len: int = GRAPHDTA_MAX_SEQ_LEN) -> np.ndarray:
    encoded = np.zeros(max_seq_len, dtype=np.int64)
    for idx, token in enumerate(str(sequence)[:max_seq_len]):
        encoded[idx] = GRAPHDTA_SEQ_DICT.get(token, 0)
    return encoded


def _one_of_k_encoding(value, allowable_set):
    if value not in allowable_set:
        raise ValueError(f"input {value!r} not in allowable set {allowable_set!r}")
    return list(map(lambda candidate: value == candidate, allowable_set))


def _one_of_k_encoding_unk(value, allowable_set):
    if value not in allowable_set:
        value = allowable_set[-1]
    return list(map(lambda candidate: value == candidate, allowable_set))


def _atom_features(atom) -> np.ndarray:
    features = np.array(
        _one_of_k_encoding_unk(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "Mg",
                "Na",
                "Ca",
                "Fe",
                "As",
                "Al",
                "I",
                "B",
                "V",
                "K",
                "Tl",
                "Yb",
                "Sb",
                "Sn",
                "Ag",
                "Pd",
                "Co",
                "Se",
                "Ti",
                "Zn",
                "H",
                "Li",
                "Ge",
                "Cu",
                "Au",
                "Ni",
                "Cd",
                "In",
                "Mn",
                "Zr",
                "Cr",
                "Pt",
                "Hg",
                "Pb",
                "Unknown",
            ],
        )
        + _one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + _one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + _one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + [atom.GetIsAromatic()]
    )
    return features / features.sum()


def smile_to_graph(smile: str) -> tuple[int, np.ndarray, list[list[int]]]:
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smile}")

    atom_count = mol.GetNumAtoms()
    features = np.asarray([_atom_features(atom) for atom in mol.GetAtoms()], dtype=np.float32)

    edge_index: list[list[int]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_index.append([begin, end])
        edge_index.append([end, begin])
    if not edge_index:
        edge_index.append([0, 0])

    return atom_count, features, edge_index


def _smile_to_graph_item(smile: str) -> tuple[str, tuple[int, np.ndarray, list[list[int]]]]:
    return smile, smile_to_graph(smile)


def build_smile_graph(smiles: Iterable[str], num_workers: int = 1) -> dict[str, tuple[int, np.ndarray, list[list[int]]]]:
    unique_smiles = sorted({str(smile) for smile in smiles if pd.notna(smile)})
    if num_workers <= 1 or len(unique_smiles) <= 1:
        return {smile: smile_to_graph(smile) for smile in unique_smiles}

    max_workers = min(num_workers, len(unique_smiles))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return dict(executor.map(_smile_to_graph_item, unique_smiles))


def frame_to_graphdta_data_list(
    frame: pd.DataFrame,
    *,
    smile_graph: dict[str, tuple[int, np.ndarray, list[list[int]]]],
    max_seq_len: int = GRAPHDTA_MAX_SEQ_LEN,
) -> list[Data]:
    required = {"sample_id", "Drug", "Target", "Y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"GraphDTA frame is missing columns: {sorted(missing)}")

    data_list: list[Data] = []
    for row in frame.itertuples(index=False):
        atom_count, features, edge_index = smile_graph[row.Drug]
        encoded_target = graphdta_seq_cat(row.Target, max_seq_len=max_seq_len)
        graph_data = Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
            y=torch.tensor([float(row.Y)], dtype=torch.float32),
        )
        graph_data.target = torch.from_numpy(np.expand_dims(encoded_target, axis=0).copy()).long()
        graph_data.c_size = torch.tensor([atom_count], dtype=torch.long)
        data_list.append(graph_data)
    return data_list


def resolve_graphdta_model(graphdta_repo: str | Path, model_name: str):
    graphdta_repo = Path(graphdta_repo).resolve()
    import sys

    if str(graphdta_repo) not in sys.path:
        sys.path.insert(0, str(graphdta_repo))
    module_name, class_name = GRAPHDTA_MODEL_SPECS[model_name]
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)

