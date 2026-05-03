from __future__ import annotations

from dataclasses import dataclass
import os

from joblib import Parallel, delayed
import numpy as np
from scipy import sparse

from mirage_mini.features import ModalFeatures


def _as_csr_float32(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    return matrix.astype(np.float32).tocsr()


def _row_l2_norms(matrix: sparse.csr_matrix) -> np.ndarray:
    squared = matrix.multiply(matrix).sum(axis=1)
    return np.sqrt(np.asarray(squared).ravel()).astype(np.float32)


def _topk_from_scores(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n_rows, n_cols = scores.shape
    if k <= 0 or n_cols == 0:
        return (
            np.empty((n_rows, 0), dtype=np.float32),
            np.empty((n_rows, 0), dtype=np.int64),
        )
    k = min(k, n_cols)
    kth = max(0, k - 1)
    topk_idx = np.argpartition(-scores, kth=kth, axis=1)[:, :k]
    topk_scores = np.take_along_axis(scores, topk_idx, axis=1)
    order = np.argsort(-topk_scores, axis=1)
    topk_idx = np.take_along_axis(topk_idx, order, axis=1)
    topk_scores = np.take_along_axis(topk_scores, order, axis=1)
    return topk_scores.astype(np.float32), topk_idx.astype(np.int64)


def _choose_reference_indices(
    n_rows: int,
    max_reference_size: int | None,
    seed: int,
) -> np.ndarray:
    if max_reference_size is None or n_rows <= max_reference_size:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = rng.choice(n_rows, size=max_reference_size, replace=False)
    return np.sort(selected.astype(np.int64))


def _resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None:
        raw = os.environ.get("MIRAGE_RETRIEVAL_N_JOBS", "").strip()
        if not raw:
            return 1
        try:
            n_jobs = int(raw)
        except ValueError:
            return 1
    return max(1, int(n_jobs))


def _iter_block_ranges(n_rows: int, chunk_size: int) -> list[tuple[int, int]]:
    size = max(1, int(chunk_size))
    return [(start, min(start + size, n_rows)) for start in range(0, n_rows, size)]


def _cosine_topk_block(
    start: int,
    stop: int,
    x_query: sparse.csr_matrix,
    x_train: sparse.csr_matrix,
    query_norms: np.ndarray,
    train_norms: np.ndarray,
    k: int,
    drop_self: bool,
    query_reference_positions: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    block = x_query[start:stop]
    scores = (block @ x_train.T).toarray().astype(np.float32)
    denom = query_norms[start:stop, None] * train_norms[None, :]
    np.divide(scores, denom, out=scores, where=denom > 0)
    if drop_self and query_reference_positions is not None:
        block_positions = query_reference_positions[start:stop]
        valid = block_positions >= 0
        if valid.any():
            rows = np.flatnonzero(valid)
            scores[rows, block_positions[valid]] = -1.0
    return _topk_from_scores(scores, k=k)


def _cosine_topk_blockwise(
    x_query: sparse.csr_matrix,
    x_train: sparse.csr_matrix,
    train_norms: np.ndarray,
    k: int,
    chunk_size: int,
    drop_self: bool,
    query_reference_positions: np.ndarray | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    x_query = _as_csr_float32(x_query)
    x_train = _as_csr_float32(x_train)
    query_norms = _row_l2_norms(x_query)
    n_query = x_query.shape[0]
    if k <= 0 or x_train.shape[0] == 0:
        return (
            np.empty((n_query, 0), dtype=np.float32),
            np.empty((n_query, 0), dtype=np.int64),
        )

    block_ranges = _iter_block_ranges(n_query, chunk_size)
    if _resolve_n_jobs(n_jobs) == 1 or len(block_ranges) <= 1:
        block_outputs = [
            _cosine_topk_block(
                start=start,
                stop=stop,
                x_query=x_query,
                x_train=x_train,
                query_norms=query_norms,
                train_norms=train_norms,
                k=k,
                drop_self=drop_self,
                query_reference_positions=query_reference_positions,
            )
            for start, stop in block_ranges
        ]
    else:
        block_outputs = Parallel(n_jobs=_resolve_n_jobs(n_jobs), prefer="threads")(
            delayed(_cosine_topk_block)(
                start=start,
                stop=stop,
                x_query=x_query,
                x_train=x_train,
                query_norms=query_norms,
                train_norms=train_norms,
                k=k,
                drop_self=drop_self,
                query_reference_positions=query_reference_positions,
            )
            for start, stop in block_ranges
        )
    sims_blocks = [block_sims for block_sims, _ in block_outputs]
    idx_blocks = [block_idx for _, block_idx in block_outputs]
    return np.vstack(sims_blocks), np.vstack(idx_blocks)


class RetrievalAugmentor:
    def __init__(
        self,
        n_neighbors: int = 8,
        chunk_size: int = 512,
        max_reference_size: int | None = None,
        reference_seed: int = 42,
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.chunk_size = max(1, int(chunk_size))
        self.max_reference_size = max_reference_size
        self.reference_seed = int(reference_seed)
        self.n_jobs = _resolve_n_jobs(n_jobs)
        self.x_train: sparse.csr_matrix | None = None
        self.train_norms: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.train_size: int = 0
        self.reference_indices: np.ndarray | None = None
        self.query_reference_positions: np.ndarray | None = None

    def fit(self, x_train: sparse.csr_matrix, y_train: np.ndarray) -> "RetrievalAugmentor":
        y_train = np.asarray(y_train)
        self.train_size = int(len(y_train))
        self.reference_indices = _choose_reference_indices(
            n_rows=self.train_size,
            max_reference_size=self.max_reference_size,
            seed=self.reference_seed,
        )
        self.query_reference_positions = np.full(self.train_size, -1, dtype=np.int64)
        self.query_reference_positions[self.reference_indices] = np.arange(len(self.reference_indices), dtype=np.int64)
        self.x_train = _as_csr_float32(x_train[self.reference_indices])
        self.train_norms = _row_l2_norms(self.x_train)
        self.y_train = y_train[self.reference_indices]
        return self

    def _summarize(self, distances: np.ndarray, indices: np.ndarray, drop_self: bool) -> np.ndarray:
        outputs = []
        for row_dist, row_idx in zip(distances, indices):
            if drop_self and len(row_idx) > 1:
                row_dist = row_dist[1:]
                row_idx = row_idx[1:]
            if len(row_idx) == 0:
                outputs.append([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            sims = 1.0 - row_dist
            labels = self.y_train[row_idx]
            outputs.append(
                [
                    float(labels.mean()),
                    float(labels.std()),
                    float(sims.mean()),
                    float(sims.max()),
                    float((labels == 1).sum()),
                ]
            )
        return np.asarray(outputs, dtype=np.float32)

    def transform_train(self, x_train: sparse.csr_matrix) -> np.ndarray:
        if self.x_train is None or self.train_norms is None:
            raise RuntimeError("RetrievalAugmentor must be fit before calling transform_train")
        sims, indices = _cosine_topk_blockwise(
            x_query=x_train,
            x_train=self.x_train,
            train_norms=self.train_norms,
            k=min(self.n_neighbors + 1, len(self.y_train)),
            chunk_size=self.chunk_size,
            drop_self=True,
            query_reference_positions=self.query_reference_positions if x_train.shape[0] == self.train_size else None,
            n_jobs=self.n_jobs,
        )
        distances = 1.0 - sims
        return self._summarize(distances, indices, drop_self=True)

    def transform(self, x_query: sparse.csr_matrix) -> np.ndarray:
        if self.x_train is None or self.train_norms is None:
            raise RuntimeError("RetrievalAugmentor must be fit before calling transform")
        sims, indices = _cosine_topk_blockwise(
            x_query=x_query,
            x_train=self.x_train,
            train_norms=self.train_norms,
            k=min(self.n_neighbors, len(self.y_train)),
            chunk_size=self.chunk_size,
            drop_self=False,
            n_jobs=self.n_jobs,
        )
        distances = 1.0 - sims
        return self._summarize(distances, indices, drop_self=False)


class RichRetrievalAugmentor:
    def __init__(
        self,
        n_neighbors: int = 8,
        chunk_size: int = 512,
        max_reference_size: int | None = None,
        reference_seed: int = 42,
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.chunk_size = max(1, int(chunk_size))
        self.max_reference_size = max_reference_size
        self.reference_seed = int(reference_seed)
        self.n_jobs = _resolve_n_jobs(n_jobs)
        self.x_train: sparse.csr_matrix | None = None
        self.train_norms: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.train_size: int = 0
        self.reference_indices: np.ndarray | None = None
        self.query_reference_positions: np.ndarray | None = None

    @property
    def stats_dim(self) -> int:
        return 6

    def fit(self, x_train: sparse.csr_matrix, y_train: np.ndarray) -> "RichRetrievalAugmentor":
        y_train = np.asarray(y_train)
        self.train_size = int(len(y_train))
        if len(y_train) == 0:
            self.x_train = None
            self.train_norms = None
            self.y_train = None
            self.reference_indices = None
            self.query_reference_positions = None
            return self
        self.reference_indices = _choose_reference_indices(
            n_rows=self.train_size,
            max_reference_size=self.max_reference_size,
            seed=self.reference_seed,
        )
        self.query_reference_positions = np.full(self.train_size, -1, dtype=np.int64)
        self.query_reference_positions[self.reference_indices] = np.arange(len(self.reference_indices), dtype=np.int64)
        self.x_train = _as_csr_float32(x_train[self.reference_indices])
        self.train_norms = _row_l2_norms(self.x_train)
        self.y_train = y_train[self.reference_indices]
        return self

    def _zero_block(self, n_rows: int) -> np.ndarray:
        return np.zeros((n_rows, self.stats_dim), dtype=np.float32)

    def _summarize(self, distances: np.ndarray, indices: np.ndarray, drop_self: bool) -> np.ndarray:
        if self.y_train is None:
            return self._zero_block(len(distances))
        outputs = []
        for row_dist, row_idx in zip(distances, indices):
            if drop_self and len(row_idx) > 1:
                row_dist = row_dist[1:]
                row_idx = row_idx[1:]
            if len(row_idx) == 0:
                outputs.append([0.0] * self.stats_dim)
                continue
            sims = np.clip(1.0 - row_dist, 0.0, 1.0)
            labels = self.y_train[row_idx]
            weights = sims + 1e-6
            weighted_mean = float(np.average(labels, weights=weights))
            outputs.append(
                [
                    float(labels.mean()),
                    float(labels.std()),
                    float(sims.mean()),
                    float(sims.max()),
                    float((labels == 1).mean()),
                    weighted_mean,
                ]
            )
        return np.asarray(outputs, dtype=np.float32)

    def transform_train(self, x_train: sparse.csr_matrix) -> np.ndarray:
        if self.x_train is None or self.train_norms is None:
            return self._zero_block(x_train.shape[0])
        sims, indices = _cosine_topk_blockwise(
            x_query=x_train,
            x_train=self.x_train,
            train_norms=self.train_norms,
            k=min(self.n_neighbors + 1, len(self.y_train)),
            chunk_size=self.chunk_size,
            drop_self=True,
            query_reference_positions=self.query_reference_positions if x_train.shape[0] == self.train_size else None,
            n_jobs=self.n_jobs,
        )
        distances = 1.0 - sims
        return self._summarize(distances, indices, drop_self=True)

    def transform(self, x_query: sparse.csr_matrix) -> np.ndarray:
        if self.x_train is None or self.train_norms is None:
            return self._zero_block(x_query.shape[0])
        sims, indices = _cosine_topk_blockwise(
            x_query=x_query,
            x_train=self.x_train,
            train_norms=self.train_norms,
            k=min(self.n_neighbors, len(self.y_train)),
            chunk_size=self.chunk_size,
            drop_self=False,
            n_jobs=self.n_jobs,
        )
        distances = 1.0 - sims
        return self._summarize(distances, indices, drop_self=False)


@dataclass
class _ViewRetrievalState:
    name: str
    augmentor: RichRetrievalAugmentor
    train_mask: np.ndarray


class MultiViewRetrievalAugmentor:
    def __init__(
        self,
        n_neighbors: int = 8,
        chunk_size: int = 512,
        max_reference_size: int | None = None,
        reference_seed: int = 42,
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.chunk_size = max(1, int(chunk_size))
        self.max_reference_size = max_reference_size
        self.reference_seed = int(reference_seed)
        self.n_jobs = _resolve_n_jobs(n_jobs)
        self.views: list[_ViewRetrievalState] = []
        self.view_names = ["joint", "smiles", "sequence", "text"]
        self.stats_dim = RichRetrievalAugmentor(
            n_neighbors=n_neighbors,
            chunk_size=chunk_size,
            max_reference_size=max_reference_size,
            reference_seed=reference_seed,
            n_jobs=self.n_jobs,
        ).stats_dim

    def fit(self, train_features: ModalFeatures, y_train: np.ndarray) -> "MultiViewRetrievalAugmentor":
        joint_mask = np.ones(len(y_train), dtype=bool)
        self.views = [
            self._fit_view("joint", train_features.matrix, joint_mask, y_train),
            self._fit_view("smiles", train_features.smiles_matrix, train_features.masks[:, 0] > 0, y_train),
            self._fit_view("sequence", train_features.sequence_matrix, train_features.masks[:, 1] > 0, y_train),
            self._fit_view("text", train_features.text_matrix, train_features.masks[:, 2] > 0, y_train),
        ]
        return self

    def _fit_view(
        self,
        name: str,
        matrix: sparse.csr_matrix,
        present_mask: np.ndarray,
        y_train: np.ndarray,
    ) -> _ViewRetrievalState:
        augmentor = RichRetrievalAugmentor(
            n_neighbors=self.n_neighbors,
            chunk_size=self.chunk_size,
            max_reference_size=self.max_reference_size,
            reference_seed=self.reference_seed,
            n_jobs=self.n_jobs,
        )
        if present_mask.sum() >= 2:
            augmentor.fit(matrix[present_mask], y_train[present_mask])
        else:
            augmentor.fit(sparse.csr_matrix((0, matrix.shape[1])), np.asarray([], dtype=np.float32))
        return _ViewRetrievalState(name=name, augmentor=augmentor, train_mask=present_mask.astype(bool))

    def _extract_view_matrix(self, features: ModalFeatures, name: str) -> sparse.csr_matrix:
        if name == "joint":
            return features.matrix
        if name == "smiles":
            return features.smiles_matrix
        if name == "sequence":
            return features.sequence_matrix
        if name == "text":
            return features.text_matrix
        raise ValueError(f"Unsupported view: {name}")

    def _query_presence_mask(self, features: ModalFeatures, name: str) -> np.ndarray:
        if name == "joint":
            return np.ones(features.matrix.shape[0], dtype=bool)
        column = {"smiles": 0, "sequence": 1, "text": 2}[name]
        return features.masks[:, column] > 0

    def _transform(self, features: ModalFeatures, is_train: bool) -> np.ndarray:
        blocks = []
        n_rows = features.matrix.shape[0]
        for view in self.views:
            block = np.zeros((n_rows, self.stats_dim), dtype=np.float32)
            query_mask = self._query_presence_mask(features, view.name)
            if query_mask.any():
                view_matrix = self._extract_view_matrix(features, view.name)[query_mask]
                if is_train and view.name != "joint":
                    block[query_mask] = view.augmentor.transform_train(view_matrix)
                elif is_train:
                    block[:] = view.augmentor.transform_train(features.matrix)
                else:
                    block[query_mask] = view.augmentor.transform(view_matrix)
            elif is_train and view.name == "joint":
                block[:] = view.augmentor.transform_train(features.matrix)
            blocks.append(block)
        return np.hstack(blocks).astype(np.float32)

    def transform_train(self, train_features: ModalFeatures) -> np.ndarray:
        return self._transform(train_features, is_train=True)

    def transform(self, query_features: ModalFeatures) -> np.ndarray:
        return self._transform(query_features, is_train=False)


class TanimotoRetrievalAugmentor:
    def __init__(
        self,
        n_neighbors: int = 8,
        chunk_size: int = 512,
        max_reference_size: int | None = None,
        reference_seed: int = 42,
        n_jobs: int | None = None,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.chunk_size = max(1, int(chunk_size))
        self.max_reference_size = max_reference_size
        self.reference_seed = int(reference_seed)
        self.n_jobs = _resolve_n_jobs(n_jobs)
        self.x_train: sparse.csr_matrix | None = None
        self.y_train: np.ndarray | None = None
        self.train_popcounts: np.ndarray | None = None
        self.stats_dim = 6
        self.train_size: int = 0
        self.reference_indices: np.ndarray | None = None
        self.query_reference_positions: np.ndarray | None = None

    def fit(self, x_train: sparse.csr_matrix, y_train: np.ndarray) -> "TanimotoRetrievalAugmentor":
        y_train = np.asarray(y_train)
        self.train_size = int(len(y_train))
        self.reference_indices = _choose_reference_indices(
            n_rows=self.train_size,
            max_reference_size=self.max_reference_size,
            seed=self.reference_seed,
        )
        self.query_reference_positions = np.full(self.train_size, -1, dtype=np.int64)
        self.query_reference_positions[self.reference_indices] = np.arange(len(self.reference_indices), dtype=np.int64)
        self.x_train = x_train[self.reference_indices].astype(np.float32).tocsr()
        self.y_train = y_train[self.reference_indices]
        self.train_popcounts = np.asarray(self.x_train.sum(axis=1)).ravel().astype(np.float32)
        return self

    def _zero_block(self, n_rows: int) -> np.ndarray:
        return np.zeros((n_rows, self.stats_dim), dtype=np.float32)

    def _tanimoto_topk(self, x_query: sparse.csr_matrix, drop_self: bool) -> tuple[np.ndarray, np.ndarray]:
        if self.x_train is None or self.y_train is None or self.train_popcounts is None:
            return np.empty((0, 0)), np.empty((0, 0), dtype=np.int64)
        x_query = _as_csr_float32(x_query)
        query_popcounts = np.asarray(x_query.sum(axis=1)).ravel().astype(np.float32)
        n_query = x_query.shape[0]
        k = min(self.n_neighbors + (1 if drop_self else 0), self.x_train.shape[0])
        if k == 0:
            return np.empty((n_query, 0), dtype=np.float32), np.empty((n_query, 0), dtype=np.int64)
        block_ranges = _iter_block_ranges(n_query, self.chunk_size)

        def _process_block(start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
            block = x_query[start:stop]
            inter = (block @ self.x_train.T).toarray().astype(np.float32)
            unions = query_popcounts[start:stop, None] + self.train_popcounts[None, :] - inter
            sims = np.divide(inter, unions, out=np.zeros_like(inter), where=unions > 0)
            if drop_self and self.query_reference_positions is not None and x_query.shape[0] == self.train_size:
                block_positions = self.query_reference_positions[start:stop]
                valid = block_positions >= 0
                if valid.any():
                    rows = np.flatnonzero(valid)
                    sims[rows, block_positions[valid]] = -1.0
            return _topk_from_scores(sims, k=k)

        if self.n_jobs == 1 or len(block_ranges) <= 1:
            block_outputs = [_process_block(start, stop) for start, stop in block_ranges]
        else:
            block_outputs = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_process_block)(start, stop) for start, stop in block_ranges
            )
        sims_blocks = [block_sims for block_sims, _ in block_outputs]
        idx_blocks = [block_idx for _, block_idx in block_outputs]
        return np.vstack(sims_blocks), np.vstack(idx_blocks)

    def _summarize(self, sims: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if self.y_train is None:
            return self._zero_block(len(sims))
        outputs = []
        for row_sims, row_idx in zip(sims, indices):
            valid = row_sims >= 0
            row_sims = row_sims[valid]
            row_idx = row_idx[valid]
            if len(row_idx) == 0:
                outputs.append([0.0] * self.stats_dim)
                continue
            labels = self.y_train[row_idx]
            weights = row_sims + 1e-6
            outputs.append(
                [
                    float(labels.mean()),
                    float(labels.std()),
                    float(row_sims.mean()),
                    float(row_sims.max()),
                    float((labels == 1).mean()),
                    float(np.average(labels, weights=weights)),
                ]
            )
        return np.asarray(outputs, dtype=np.float32)

    def transform_train(self, x_train: sparse.csr_matrix) -> np.ndarray:
        sims, idx = self._tanimoto_topk(x_train, drop_self=True)
        return self._summarize(sims, idx)

    def transform(self, x_query: sparse.csr_matrix) -> np.ndarray:
        sims, idx = self._tanimoto_topk(x_query, drop_self=False)
        return self._summarize(sims, idx)

