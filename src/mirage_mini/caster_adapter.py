from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .external_baselines import prepare_external_regression_frame


CASTER_SPLIT_COLUMNS = [
    "protein_id",
    "protein_sequence",
    "molecule_id",
    "molecule_smiles",
    "affinity_score",
    "binary_label",
    "sample_id",
    "drug_id",
    "target_id",
    "label_raw_nm",
]


def prepare_caster_split_frame(
    frame: pd.DataFrame,
    *,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    external = prepare_external_regression_frame(frame, dataset_name=dataset_name)
    converted = pd.DataFrame(
        {
            "protein_id": external["target_id"].astype(str),
            "protein_sequence": external["Target"].astype(str),
            "molecule_id": external["drug_id"].astype(str),
            "molecule_smiles": external["Drug"].astype(str),
            "affinity_score": external["Y"].astype(float),
            "binary_label": external["binary_label"].astype(int),
            "sample_id": external["sample_id"].astype(str),
            "drug_id": external["drug_id"].astype(str),
            "target_id": external["target_id"].astype(str),
            "label_raw_nm": external["label_raw_nm"].astype(float),
        }
    )
    return converted[CASTER_SPLIT_COLUMNS].reset_index(drop=True)


def build_caster_matrix_inputs(
    split_frames: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
) -> tuple[OrderedDict[str, str], OrderedDict[str, str], np.ndarray]:
    if isinstance(split_frames, Mapping):
        frames = list(split_frames.values())
    else:
        frames = list(split_frames)
    if not frames:
        raise ValueError("At least one CASTER split frame is required.")

    merged = pd.concat(frames, ignore_index=True)

    protein_info = (
        merged[["protein_id", "protein_sequence"]]
        .drop_duplicates(subset=["protein_id"], keep="first")
        .sort_values("protein_id")
        .reset_index(drop=True)
    )
    molecule_info = (
        merged[["molecule_id", "molecule_smiles"]]
        .drop_duplicates(subset=["molecule_id"], keep="first")
        .sort_values("molecule_id")
        .reset_index(drop=True)
    )

    proteins = OrderedDict(
        (row.protein_id, row.protein_sequence)
        for row in protein_info.itertuples(index=False)
    )
    ligands = OrderedDict(
        (row.molecule_id, row.molecule_smiles)
        for row in molecule_info.itertuples(index=False)
    )

    protein_to_index = {protein_id: idx for idx, protein_id in enumerate(proteins.keys())}
    ligand_to_index = {molecule_id: idx for idx, molecule_id in enumerate(ligands.keys())}

    affinity = np.full((len(ligands), len(proteins)), np.nan, dtype=float)
    for row in merged.itertuples(index=False):
        affinity[ligand_to_index[row.molecule_id], protein_to_index[row.protein_id]] = float(row.affinity_score)

    return proteins, ligands, affinity

