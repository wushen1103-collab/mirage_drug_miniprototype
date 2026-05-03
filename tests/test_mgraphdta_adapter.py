from __future__ import annotations

import numpy as np
import pandas as pd

from mirage_mini.mgraphdta_adapter import (
    MGRAPHDTA_MAX_SEQ_LEN,
    MGRAPHDTA_SEQ_DICT,
    build_mgraph_smile_graph,
    frame_to_mgraphdta_data_list,
    mgraphdta_seq_cat,
    mgraph_smile_to_graph,
)


def test_mgraphdta_seq_cat_maps_known_tokens_unknowns_and_padding():
    encoded = mgraphdta_seq_cat("ACZ?", max_seq_len=6)
    assert encoded.tolist() == [
        MGRAPHDTA_SEQ_DICT["A"],
        MGRAPHDTA_SEQ_DICT["C"],
        MGRAPHDTA_SEQ_DICT["Z"],
        0,
        0,
        0,
    ]


def test_mgraphdta_seq_cat_truncates_to_max_length():
    encoded = mgraphdta_seq_cat("ACBED", max_seq_len=4)
    assert encoded.tolist() == [
        MGRAPHDTA_SEQ_DICT["A"],
        MGRAPHDTA_SEQ_DICT["C"],
        MGRAPHDTA_SEQ_DICT["B"],
        MGRAPHDTA_SEQ_DICT["E"],
    ]


def test_mgraph_smile_to_graph_returns_expected_shapes():
    node_features, edge_index, edge_attr = mgraph_smile_to_graph("CCO")
    assert node_features.shape == (3, 22)
    assert edge_index.shape[0] == 2
    assert edge_attr.shape[1] == 6
    assert np.all(np.isfinite(node_features))
    assert len(edge_index.T) >= 4


def test_build_mgraph_smile_graph_deduplicates_inputs():
    smile_graph = build_mgraph_smile_graph(["CCO", "CCO", "CCN"], num_workers=1)
    assert set(smile_graph) == {"CCO", "CCN"}


def test_frame_to_mgraphdta_data_list_builds_pyg_examples():
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "Drug": ["CCO", "CCN"],
            "Target": ["ACD", "KLM"],
            "Y": [7.2, 6.5],
        }
    )
    smile_graph = build_mgraph_smile_graph(frame["Drug"], num_workers=1)
    data_list = frame_to_mgraphdta_data_list(frame, smile_graph=smile_graph)

    assert len(data_list) == 2
    assert tuple(data_list[0].x.shape) == (3, 22)
    assert tuple(data_list[0].edge_attr.shape)[1] == 6
    assert tuple(data_list[0].target.shape) == (1, MGRAPHDTA_MAX_SEQ_LEN)
    assert tuple(data_list[0].y.shape) == (1,)

