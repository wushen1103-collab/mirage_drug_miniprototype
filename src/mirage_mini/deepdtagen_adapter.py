from __future__ import annotations

import pandas as pd

from .external_baselines import prepare_external_regression_frame


DEEPDTAGEN_COLUMNS = [
    "compound_iso_smiles",
    "target_smiles",
    "target_sequence",
    "affinity",
    "binary_label",
    "sample_id",
    "drug_id",
    "target_id",
    "label_raw_nm",
]


def prepare_deepdtagen_split_frame(
    frame: pd.DataFrame,
    *,
    target_smiles_mode: str = "self",
    dataset_name: str | None = None,
) -> pd.DataFrame:
    if target_smiles_mode != "self":
        raise ValueError(f"Unsupported target_smiles_mode: {target_smiles_mode}")

    external = prepare_external_regression_frame(frame, dataset_name=dataset_name)
    converted = pd.DataFrame(
        {
            "compound_iso_smiles": external["Drug"].astype(str),
            "target_smiles": external["Drug"].astype(str),
            "target_sequence": external["Target"].astype(str),
            "affinity": external["Y"].astype(float),
            "binary_label": external["binary_label"].astype(int),
            "sample_id": external["sample_id"].astype(str),
            "drug_id": external["drug_id"].astype(str),
            "target_id": external["target_id"].astype(str),
            "label_raw_nm": external["label_raw_nm"].astype(float),
        }
    )
    return converted[DEEPDTAGEN_COLUMNS].reset_index(drop=True)

