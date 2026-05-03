from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from .external_baselines import prepare_external_regression_frame


KANPM_SPLIT_COLUMNS = [
    "drug_key",
    "compound_iso_smiles",
    "target_key",
    "target_sequence",
    "affinity",
    "binary_label",
    "sample_id",
    "drug_id",
    "target_id",
    "label_raw_nm",
]


def prepare_kanpm_split_frame(
    frame: pd.DataFrame,
    *,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    external = prepare_external_regression_frame(frame, dataset_name=dataset_name)
    converted = pd.DataFrame(
        {
            "drug_key": external["drug_id"].astype(str),
            "compound_iso_smiles": external["Drug"].astype(str),
            "target_key": external["target_id"].astype(str),
            "target_sequence": external["Target"].astype(str),
            "affinity": external["Y"].astype(float),
            "binary_label": external["binary_label"].astype(int),
            "sample_id": external["sample_id"].astype(str),
            "drug_id": external["drug_id"].astype(str),
            "target_id": external["target_id"].astype(str),
            "label_raw_nm": external["label_raw_nm"].astype(float),
        }
    )
    return converted[KANPM_SPLIT_COLUMNS].reset_index(drop=True)


def build_kanpm_entity_tables(
    split_frames: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(split_frames, Mapping):
        frames = list(split_frames.values())
    else:
        frames = list(split_frames)
    if not frames:
        raise ValueError("At least one KANPM split frame is required.")

    merged = pd.concat(frames, ignore_index=True)

    drugs = (
        merged[["drug_key", "compound_iso_smiles"]]
        .drop_duplicates(subset=["drug_key"], keep="first")
        .reset_index(drop=True)
    )
    drugs["drug_id"] = drugs["drug_key"].astype(str)
    drugs["drug_seq"] = drugs["compound_iso_smiles"].astype(str)
    drugs = drugs[["drug_key", "compound_iso_smiles", "drug_id", "drug_seq"]]

    proteins = (
        merged[["target_key", "target_sequence"]]
        .drop_duplicates(subset=["target_key"], keep="first")
        .reset_index(drop=True)
    )
    proteins["prot_id"] = proteins["target_key"].astype(str)
    proteins["prot_seq"] = proteins["target_sequence"].astype(str)
    proteins = proteins[["target_key", "target_sequence", "prot_id", "prot_seq"]]

    return drugs, proteins

