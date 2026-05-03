from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from importlib import util
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from torch_geometric.data import Data


_FDEF_NAME = Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"
_CHEM_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(str(_FDEF_NAME))

MGRAPHDTA_SEQ_DICT = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}
MGRAPHDTA_MAX_SEQ_LEN = 1200

_ATOM_TYPES = ["H", "C", "N", "O", "F", "Cl", "S", "Br", "I"]
_HYBRIDIZATION_TYPES = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def mgraphdta_seq_cat(sequence: str, max_seq_len: int = MGRAPHDTA_MAX_SEQ_LEN) -> np.ndarray:
    encoded = np.zeros(max_seq_len, dtype=np.int64)
    for idx, token in enumerate(str(sequence)[:max_seq_len]):
        encoded[idx] = MGRAPHDTA_SEQ_DICT.get(token, 0)
    return encoded


def _atom_feature_vector(atom, *, acceptor: int, donor: int) -> list[float]:
    features: list[float] = []
    features.extend(int(atom.GetSymbol() == atom_type) for atom_type in _ATOM_TYPES)
    features.append(float(atom.GetAtomicNum()))
    features.append(float(acceptor))
    features.append(float(donor))
    features.append(float(atom.GetIsAromatic()))
    features.extend(float(atom.GetHybridization() == hybridization) for hybridization in _HYBRIDIZATION_TYPES)
    features.append(float(atom.GetTotalNumHs()))
    features.append(float(atom.GetExplicitValence()))
    features.append(float(atom.GetFormalCharge()))
    features.append(float(atom.GetImplicitValence()))
    features.append(float(atom.GetNumExplicitHs()))
    features.append(float(atom.GetNumRadicalElectrons()))
    return features


def _bond_feature_vector(bond) -> list[float]:
    features = [float(bond.GetBondType() == bond_type) for bond_type in _BOND_TYPES]
    is_conjugated = float(bond.GetIsConjugated())
    features.append(float(not is_conjugated))
    features.append(is_conjugated)
    return features


def _minmax_normalize(features: np.ndarray) -> np.ndarray:
    minimum = float(features.min())
    maximum = float(features.max())
    if maximum <= minimum:
        return features.astype(np.float32, copy=False)
    return ((features - minimum) / (maximum - minimum)).astype(np.float32, copy=False)


def mgraph_smile_to_graph(smile: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smile}")

    acceptor_flags = np.zeros(mol.GetNumAtoms(), dtype=np.int64)
    donor_flags = np.zeros(mol.GetNumAtoms(), dtype=np.int64)
    for feature in _CHEM_FEATURE_FACTORY.GetFeaturesForMol(mol):
        atom_ids = feature.GetAtomIds()
        if feature.GetFamily() == "Donor":
            donor_flags[list(atom_ids)] = 1
        elif feature.GetFamily() == "Acceptor":
            acceptor_flags[list(atom_ids)] = 1

    node_features = np.asarray(
        [
            _atom_feature_vector(
                atom,
                acceptor=int(acceptor_flags[atom.GetIdx()]),
                donor=int(donor_flags[atom.GetIdx()]),
            )
            for atom in mol.GetAtoms()
        ],
        dtype=np.float32,
    )
    node_features = _minmax_normalize(node_features)

    edge_index: list[list[int]] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_features = _bond_feature_vector(bond)
        edge_index.append([begin, end])
        edge_attr.append(bond_features)
        edge_index.append([end, begin])
        edge_attr.append(bond_features)

    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    return (
        node_features,
        np.asarray(edge_index, dtype=np.int64).T,
        np.asarray(edge_attr, dtype=np.float32),
    )


def _smile_to_graph_item(smile: str) -> tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    return smile, mgraph_smile_to_graph(smile)


def build_mgraph_smile_graph(
    smiles: Iterable[str],
    num_workers: int = 1,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    unique_smiles = sorted({str(smile) for smile in smiles if pd.notna(smile)})
    if num_workers <= 1 or len(unique_smiles) <= 1:
        return {smile: mgraph_smile_to_graph(smile) for smile in unique_smiles}

    max_workers = min(num_workers, len(unique_smiles))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return dict(executor.map(_smile_to_graph_item, unique_smiles))


def frame_to_mgraphdta_data_list(
    frame: pd.DataFrame,
    *,
    smile_graph: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    max_seq_len: int = MGRAPHDTA_MAX_SEQ_LEN,
) -> list[Data]:
    required = {"sample_id", "Drug", "Target", "Y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"MGraphDTA frame is missing columns: {sorted(missing)}")

    data_list: list[Data] = []
    for row in frame.itertuples(index=False):
        node_features, edge_index, edge_attr = smile_graph[row.Drug]
        encoded_target = mgraphdta_seq_cat(row.Target, max_seq_len=max_seq_len)

        graph_data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
            y=torch.tensor([float(row.Y)], dtype=torch.float32),
        )
        graph_data.target = torch.from_numpy(np.expand_dims(encoded_target, axis=0).copy()).long()
        data_list.append(graph_data)
    return data_list


def resolve_mgraphdta_model(mgraphdta_repo: str | Path):
    repo_root = Path(mgraphdta_repo).resolve()
    model_path = repo_root / "regression" / "model.py"
    if not model_path.exists():
        raise FileNotFoundError(f"MGraphDTA model file not found: {model_path}")

    spec = util.spec_from_file_location("vendor_mgraphdta_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load MGraphDTA model from {model_path}")

    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MGraphDTA

