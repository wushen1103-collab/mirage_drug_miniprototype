from __future__ import annotations

import pandas as pd
import torch

from mirage_mini.deepdta_adapter import (
    DEEPDTA_MAX_PROTEIN_LEN,
    DEEPDTA_MAX_SMILES_LEN,
    DEEPDTA_PROTEIN_DICT,
    DEEPDTA_SMILES_DICT,
    DeepDTABatch,
    DeepDTARegressor,
    build_deepdta_batch,
    encode_deepdta_protein,
    encode_deepdta_smiles,
)


def test_encode_deepdta_smiles_maps_known_tokens_unknowns_and_padding():
    encoded = encode_deepdta_smiles("C=O?", max_len=6)
    assert encoded.tolist() == [
        DEEPDTA_SMILES_DICT["C"],
        DEEPDTA_SMILES_DICT["="],
        DEEPDTA_SMILES_DICT["O"],
        0,
        0,
        0,
    ]


def test_encode_deepdta_protein_maps_known_tokens_unknowns_and_padding():
    encoded = encode_deepdta_protein("ACZ?", max_len=6)
    assert encoded.tolist() == [
        DEEPDTA_PROTEIN_DICT["A"],
        DEEPDTA_PROTEIN_DICT["C"],
        DEEPDTA_PROTEIN_DICT["Z"],
        0,
        0,
        0,
    ]


def test_build_deepdta_batch_creates_expected_tensor_shapes():
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "Drug": ["CCO", "CCN"],
            "Target": ["ACD", "KLM"],
            "Y": [7.2, 6.5],
        }
    )

    batch = build_deepdta_batch(frame)

    assert isinstance(batch, DeepDTABatch)
    assert tuple(batch.smiles.shape) == (2, DEEPDTA_MAX_SMILES_LEN)
    assert tuple(batch.proteins.shape) == (2, DEEPDTA_MAX_PROTEIN_LEN)
    assert tuple(batch.labels.shape) == (2,)


def test_deepdta_regressor_forward_returns_batch_vector():
    model = DeepDTARegressor()
    batch = DeepDTABatch(
        smiles=torch.randint(0, len(DEEPDTA_SMILES_DICT) + 1, (4, DEEPDTA_MAX_SMILES_LEN), dtype=torch.long),
        proteins=torch.randint(0, len(DEEPDTA_PROTEIN_DICT) + 1, (4, DEEPDTA_MAX_PROTEIN_LEN), dtype=torch.long),
        labels=torch.randn(4),
    )

    prediction = model(batch.smiles, batch.proteins)

    assert tuple(prediction.shape) == (4,)

