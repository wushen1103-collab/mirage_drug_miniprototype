from __future__ import annotations

import numpy as np
import pandas as pd

from mirage_mini.graphdta_adapter import (
    build_smile_graph,
    frame_to_graphdta_data_list,
    graphdta_seq_cat,
    smile_to_graph,
)


def test_graphdta_seq_cat_maps_known_tokens_unknowns_and_padding():
    encoded = graphdta_seq_cat("ABZ?", max_seq_len=6)
    assert encoded.tolist() == [1, 2, 25, 0, 0, 0]


def test_graphdta_seq_cat_truncates_to_max_length():
    encoded = graphdta_seq_cat("ABCDEFG", max_seq_len=4)
    assert encoded.tolist() == [1, 2, 3, 4]


def test_smile_to_graph_returns_expected_basic_shapes():
    c_size, features, edge_index = smile_to_graph("CCO")
    assert c_size == 3
    assert features.shape[0] == 3
    assert features.shape[1] > 10
    assert np.allclose(features.sum(axis=1), 1.0)
    assert len(edge_index) >= 4


def test_build_smile_graph_deduplicates_inputs():
    smile_graph = build_smile_graph(["CCO", "CCO", "CCN"], num_workers=1)
    assert set(smile_graph) == {"CCO", "CCN"}


def test_frame_to_graphdta_data_list_builds_pyg_examples():
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "Drug": ["CCO", "CCN"],
            "Target": ["ACD", "KLM"],
            "Y": [7.2, 6.5],
        }
    )
    smile_graph = build_smile_graph(frame["Drug"], num_workers=1)
    data_list = frame_to_graphdta_data_list(frame, smile_graph=smile_graph, max_seq_len=8)

    assert len(data_list) == 2
    assert tuple(data_list[0].x.shape)[0] == 3
    assert tuple(data_list[0].target.shape) == (1, 8)
    assert tuple(data_list[0].y.shape) == (1,)

