from __future__ import annotations

import pandas as pd
import pytest

from mirage_mini.kanpm_adapter import build_kanpm_entity_tables, prepare_kanpm_split_frame


def test_prepare_kanpm_split_frame_converts_to_expected_columns_and_pkd():
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "drug_id": ["d1", "d2"],
            "target_id": ["t1", "t2"],
            "smiles": ["CCO", "CCC"],
            "sequence": ["MKT", "AAAA"],
            "label": [1, 0],
            "label_raw": [1000.0, 10000.0],
        }
    )

    converted = prepare_kanpm_split_frame(frame)

    assert list(converted.columns[:5]) == [
        "drug_key",
        "compound_iso_smiles",
        "target_key",
        "target_sequence",
        "affinity",
    ]
    assert converted.loc[0, "drug_key"] == "d1"
    assert converted.loc[0, "compound_iso_smiles"] == "CCO"
    assert converted.loc[0, "target_key"] == "t1"
    assert converted.loc[0, "affinity"] == pytest.approx(6.0)
    assert converted.loc[1, "affinity"] == pytest.approx(5.0)


def test_build_kanpm_entity_tables_deduplicates_drugs_and_targets():
    split_a = pd.DataFrame(
        {
            "drug_key": ["d1", "d2"],
            "compound_iso_smiles": ["CCO", "CCC"],
            "target_key": ["t1", "t1"],
            "target_sequence": ["MKT", "MKT"],
            "affinity": [6.0, 5.5],
        }
    )
    split_b = pd.DataFrame(
        {
            "drug_key": ["d2", "d3"],
            "compound_iso_smiles": ["CCC", "CCN"],
            "target_key": ["t2", "t3"],
            "target_sequence": ["AAAA", "GGGG"],
            "affinity": [5.2, 7.1],
        }
    )

    drugs, proteins = build_kanpm_entity_tables({"train": split_a, "test": split_b})

    assert list(drugs["drug_key"]) == ["d1", "d2", "d3"]
    assert list(drugs["drug_id"]) == ["d1", "d2", "d3"]
    assert list(drugs["drug_seq"]) == ["CCO", "CCC", "CCN"]
    assert list(proteins["target_key"]) == ["t1", "t2", "t3"]
    assert list(proteins["prot_id"]) == ["t1", "t2", "t3"]
    assert list(proteins["prot_seq"]) == ["MKT", "AAAA", "GGGG"]

