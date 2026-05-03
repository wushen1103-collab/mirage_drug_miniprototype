from __future__ import annotations

import pandas as pd

from mirage_mini.mixingdta_adapter import (
    build_mixingdta_entity_tables,
    frame_to_mixingdta_records,
    prepare_mixingdta_split_frame,
)


def test_prepare_mixingdta_split_frame_converts_columns():
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

    converted = prepare_mixingdta_split_frame(frame)

    assert list(converted.columns) == [
        "drug_id",
        "target_id",
        "compound_iso_smiles",
        "target_sequence",
        "affinity",
        "binary_label",
        "sample_id",
        "label_raw_nm",
    ]
    assert converted.loc[0, "affinity"] == 6.0


def test_build_mixingdta_entity_tables_and_records():
    split = pd.DataFrame(
        {
            "drug_id": ["d1", "d2"],
            "target_id": ["t1", "t2"],
            "compound_iso_smiles": ["CCO", "CCC"],
            "target_sequence": ["AAA", "BBB"],
            "affinity": [6.0, 5.0],
            "binary_label": [1, 0],
            "sample_id": ["s1", "s2"],
            "label_raw_nm": [1000.0, 10000.0],
        }
    )

    drugs, proteins = build_mixingdta_entity_tables([split])
    records = frame_to_mixingdta_records(split)

    assert list(drugs.items()) == [("d1", "CCO"), ("d2", "CCC")]
    assert list(proteins.items()) == [("t1", "AAA"), ("t2", "BBB")]
    assert records[0] == ("d1", "CCO", "t1", "AAA", 6.0)

