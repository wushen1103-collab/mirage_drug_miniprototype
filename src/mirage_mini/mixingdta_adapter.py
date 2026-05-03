from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence

import pandas as pd

from .external_baselines import prepare_external_regression_frame


MIXINGDTA_SPLIT_COLUMNS = [
    "drug_id",
    "target_id",
    "compound_iso_smiles",
    "target_sequence",
    "affinity",
    "binary_label",
    "sample_id",
    "label_raw_nm",
]


def prepare_mixingdta_split_frame(
    frame: pd.DataFrame,
    *,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    external = prepare_external_regression_frame(frame, dataset_name=dataset_name)
    converted = pd.DataFrame(
        {
            "drug_id": external["drug_id"].astype(str),
            "target_id": external["target_id"].astype(str),
            "compound_iso_smiles": external["Drug"].astype(str),
            "target_sequence": external["Target"].astype(str),
            "affinity": external["Y"].astype(float),
            "binary_label": external["binary_label"].astype(int),
            "sample_id": external["sample_id"].astype(str),
            "label_raw_nm": external["label_raw_nm"].astype(float),
        }
    )
    return converted[MIXINGDTA_SPLIT_COLUMNS].reset_index(drop=True)


def build_mixingdta_entity_tables(
    split_frames: Mapping[str, pd.DataFrame] | Sequence[pd.DataFrame],
) -> tuple[OrderedDict[str, str], OrderedDict[str, str]]:
    if isinstance(split_frames, Mapping):
        frames = list(split_frames.values())
    else:
        frames = list(split_frames)
    if not frames:
        raise ValueError("At least one MixingDTA split frame is required.")

    merged = pd.concat(frames, ignore_index=True)

    drugs = OrderedDict(
        (row.drug_id, row.compound_iso_smiles)
        for row in (
            merged[["drug_id", "compound_iso_smiles"]]
            .drop_duplicates(subset=["drug_id"], keep="first")
            .sort_values("drug_id")
            .itertuples(index=False)
        )
    )
    proteins = OrderedDict(
        (row.target_id, row.target_sequence)
        for row in (
            merged[["target_id", "target_sequence"]]
            .drop_duplicates(subset=["target_id"], keep="first")
            .sort_values("target_id")
            .itertuples(index=False)
        )
    )
    return drugs, proteins


def frame_to_mixingdta_records(frame: pd.DataFrame) -> list[tuple[str, str, str, str, float]]:
    return [
        (
            str(row.drug_id),
            str(row.compound_iso_smiles),
            str(row.target_id),
            str(row.target_sequence),
            float(row.affinity),
        )
        for row in frame.itertuples(index=False)
    ]

