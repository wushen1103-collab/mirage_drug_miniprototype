from __future__ import annotations

import numpy as np
from scipy import sparse

from mirage_mini.retrieval import RetrievalAugmentor, RichRetrievalAugmentor


def _three_point_matrix() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        np.asarray(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )


def test_retrieval_train_features_keep_the_nearest_legal_neighbor():
    matrix = _three_point_matrix()
    labels = np.asarray([0, 1, 0], dtype=np.int64)
    augmentor = RetrievalAugmentor(n_neighbors=1).fit(matrix, labels)

    stats = augmentor.transform_train(matrix)

    # For query 0, self is excluded and query 1 is the nearest legal neighbor.
    assert stats[0, 0] == 1.0
    assert stats[0, 4] == 1.0


def test_rich_retrieval_train_features_keep_the_nearest_legal_neighbor():
    matrix = _three_point_matrix()
    labels = np.asarray([0, 1, 0], dtype=np.int64)
    augmentor = RichRetrievalAugmentor(n_neighbors=1).fit(matrix, labels)

    stats = augmentor.transform_train(matrix)

    assert stats[0, 0] == 1.0
    assert stats[0, 4] == 1.0


def test_self_is_not_retained_when_only_one_legal_neighbor_exists():
    matrix = sparse.csr_matrix(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    labels = np.asarray([1, 0], dtype=np.int64)
    augmentor = RetrievalAugmentor(n_neighbors=2).fit(matrix, labels)

    stats = augmentor.transform_train(matrix)

    assert stats[0, 0] == 0.0
    assert stats[1, 0] == 1.0
