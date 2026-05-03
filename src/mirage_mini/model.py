from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def train_classifier(x_train: sparse.csr_matrix, y_train: np.ndarray) -> CalibratedClassifierCV:
    base = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        solver="liblinear",
    )
    _, counts = np.unique(y_train, return_counts=True)
    cv = max(2, min(3, int(counts.min())))
    model = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
    model.fit(x_train, y_train)
    return model


def append_dense_features(x_sparse: sparse.csr_matrix, dense: np.ndarray) -> sparse.csr_matrix:
    dense_x = sparse.csr_matrix(dense)
    return sparse.hstack([x_sparse, dense_x], format="csr")


def build_gate_features(
    mask_prob: np.ndarray,
    retrieval_prob: np.ndarray,
    retrieval_stats: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    missing_count = (1.0 - masks).sum(axis=1, keepdims=True)
    prob_gap = np.abs(retrieval_prob - mask_prob).reshape(-1, 1)
    signed_gap = (retrieval_prob - mask_prob).reshape(-1, 1)
    return np.hstack(
        [
            mask_prob.reshape(-1, 1),
            retrieval_prob.reshape(-1, 1),
            signed_gap,
            prob_gap,
            retrieval_stats,
            masks.astype(np.float32),
            missing_count.astype(np.float32),
        ]
    ).astype(np.float32)


@dataclass
class GateModel:
    model: LogisticRegression | None
    fallback_alpha: float = 0.5

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return self.model.predict_proba(x)[:, 1]
        base_prob = x[:, 0]
        retrieval_prob = x[:, 1]
        return ((1.0 - self.fallback_alpha) * base_prob + self.fallback_alpha * retrieval_prob).astype(np.float32)


def fit_gate_model(x_train: np.ndarray, y_train: np.ndarray) -> GateModel:
    y_train = np.asarray(y_train)
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or int(counts.min()) < 2 or len(y_train) < 6:
        return GateModel(model=None, fallback_alpha=0.5)
    model = LogisticRegression(
        max_iter=300,
        class_weight="balanced",
        solver="liblinear",
        C=0.5,
    )
    model.fit(x_train, y_train)
    return GateModel(model=model)


def blend_probabilities(prob_a: np.ndarray, prob_b: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return ((1.0 - alpha) * np.asarray(prob_a) + alpha * np.asarray(prob_b)).astype(np.float32)


def select_blend_weight(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    steps: int = 21,
) -> float:
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_a, dtype=np.float32)
    prob_b = np.asarray(prob_b, dtype=np.float32)
    candidates = np.linspace(0.0, 1.0, max(2, steps))
    best_alpha = 0.5
    best_score = float("-inf")
    for alpha in candidates:
        blended = blend_probabilities(prob_a, prob_b, float(alpha))
        score = float(average_precision_score(y_true, blended))
        tie_break = -abs(float(alpha) - 0.5)
        if score > best_score or (np.isclose(score, best_score) and tie_break > -abs(best_alpha - 0.5)):
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def select_best_candidate(y_true: np.ndarray, candidates: dict[str, np.ndarray]) -> str:
    best_name = next(iter(candidates))
    best_score = float("-inf")
    for name, probs in candidates.items():
        score = float(average_precision_score(y_true, np.asarray(probs, dtype=np.float32)))
        if score > best_score:
            best_score = score
            best_name = name
    return best_name
