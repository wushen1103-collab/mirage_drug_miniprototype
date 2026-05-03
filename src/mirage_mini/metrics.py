from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (prob >= left) & (prob <= right)
        else:
            mask = (prob >= left) & (prob < right)
        if not mask.any():
            continue
        accuracy = y_true[mask].mean()
        confidence = prob[mask].mean()
        ece += abs(accuracy - confidence) * (mask.sum() / len(y_true))
    return float(ece)


def risk_at_coverage(y_true: np.ndarray, prob: np.ndarray, coverage: float = 0.8) -> float:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    confidence = np.maximum(prob, 1.0 - prob)
    keep = max(1, int(len(prob) * coverage))
    selected = np.argsort(-confidence)[:keep]
    preds = (prob[selected] >= 0.5).astype(int)
    risk = 1.0 - (preds == y_true[selected]).mean()
    return float(risk)


def calibration_table(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for bin_id, (left, right) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        if right == 1.0:
            mask = (prob >= left) & (prob <= right)
        else:
            mask = (prob >= left) & (prob < right)
        if not mask.any():
            continue
        accuracy = float(y_true[mask].mean())
        mean_confidence = float(prob[mask].mean())
        gap = abs(accuracy - mean_confidence)
        count = int(mask.sum())
        rows.append(
            {
                "bin_id": bin_id,
                "bin_left": float(left),
                "bin_right": float(right),
                "count": count,
                "fraction": float(count / len(y_true)),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
                "gap": float(gap),
                "weighted_gap": float(gap * (count / len(y_true))),
            }
        )
    return pd.DataFrame(rows)


def risk_coverage_table(y_true: np.ndarray, prob: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    prob = np.asarray(prob)
    confidence = np.maximum(prob, 1.0 - prob)
    preds = (prob >= 0.5).astype(int)
    order = np.argsort(-confidence)
    sorted_true = y_true[order]
    sorted_pred = preds[order]
    sorted_confidence = confidence[order]
    correct = (sorted_true == sorted_pred).astype(int)

    rows = []
    running_correct = 0
    n = len(sorted_true)
    for idx in range(n):
        running_correct += int(correct[idx])
        selected = idx + 1
        accuracy = running_correct / selected
        rows.append(
            {
                "selected": selected,
                "coverage": float(selected / n),
                "accuracy": float(accuracy),
                "risk": float(1.0 - accuracy),
                "threshold_confidence": float(sorted_confidence[idx]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_binary(y_true: np.ndarray, prob: np.ndarray) -> dict:
    try:
        auroc = float(roc_auc_score(y_true, prob))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, prob))
    except ValueError:
        auprc = float("nan")
    return {
        "auroc": auroc,
        "auprc": auprc,
        "ece": expected_calibration_error(y_true, prob),
        "risk_at_80_coverage": risk_at_coverage(y_true, prob, coverage=0.8),
    }

