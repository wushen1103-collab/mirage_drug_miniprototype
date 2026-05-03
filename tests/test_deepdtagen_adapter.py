from __future__ import annotations

import pandas as pd
import pytest

from mirage_mini.deepdtagen_adapter import prepare_deepdtagen_split_frame


def test_prepare_deepdtagen_split_frame_copies_smiles_into_target_smiles():
    frame = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "drug_id": ["d1"],
            "target_id": ["t1"],
            "smiles": ["CCO"],
            "sequence": ["MKT"],
            "label": [1],
            "label_raw": [1000.0],
        }
    )

    converted = prepare_deepdtagen_split_frame(frame)

    assert list(converted.columns[:4]) == [
        "compound_iso_smiles",
        "target_smiles",
        "target_sequence",
        "affinity",
    ]
    assert converted.loc[0, "compound_iso_smiles"] == "CCO"
    assert converted.loc[0, "target_smiles"] == "CCO"
    assert converted.loc[0, "affinity"] == pytest.approx(6.0)


def test_prepare_deepdtagen_split_frame_rejects_unknown_target_smiles_mode():
    frame = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "drug_id": ["d1"],
            "target_id": ["t1"],
            "smiles": ["CCO"],
            "sequence": ["MKT"],
            "label": [1],
            "label_raw": [1000.0],
        }
    )

    with pytest.raises(ValueError, match="Unsupported target_smiles_mode"):
        prepare_deepdtagen_split_frame(frame, target_smiles_mode="unknown")

