from __future__ import annotations

import numpy as np
import pandas as pd

from mirage_mini.caster_adapter import build_caster_matrix_inputs, prepare_caster_split_frame


def test_prepare_caster_split_frame_converts_columns():
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

    converted = prepare_caster_split_frame(frame)

    assert list(converted.columns) == [
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
    assert converted.loc[0, "protein_id"] == "t1"
    assert converted.loc[0, "molecule_id"] == "d1"
    assert converted.loc[0, "affinity_score"] == 6.0


def test_build_caster_matrix_inputs_populates_sparse_affinity_matrix():
    split = pd.DataFrame(
        {
            "protein_id": ["t1", "t2"],
            "protein_sequence": ["AAA", "BBB"],
            "molecule_id": ["d1", "d2"],
            "molecule_smiles": ["CCO", "CCC"],
            "affinity_score": [6.0, 5.0],
            "binary_label": [1, 0],
            "sample_id": ["s1", "s2"],
            "drug_id": ["d1", "d2"],
            "target_id": ["t1", "t2"],
            "label_raw_nm": [1000.0, 10000.0],
        }
    )

    proteins, ligands, affinity = build_caster_matrix_inputs([split])

    assert list(proteins.keys()) == ["t1", "t2"]
    assert list(ligands.keys()) == ["d1", "d2"]
    assert affinity.shape == (2, 2)
    assert affinity[0, 0] == 6.0
    assert affinity[1, 1] == 5.0
    assert np.isnan(affinity[0, 1])

