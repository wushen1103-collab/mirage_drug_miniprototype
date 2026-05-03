from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REQUIRED_EXTERNAL_FRAME_COLUMNS = {
    "sample_id",
    "drug_id",
    "target_id",
    "smiles",
    "sequence",
    "label",
    "label_raw",
}


def nm_to_pkd(values):
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0):
        raise ValueError("Affinity values must be strictly positive to convert nM to pKd.")
    transformed = 9.0 - np.log10(array)
    if np.isscalar(values):
        return float(transformed)
    return transformed


def _uses_raw_affinity_scale(dataset_name: str | None) -> bool:
    normalized = str(dataset_name or "").upper().strip()
    return normalized == "KIBA"


def affinity_to_regression_target(values, dataset_name: str | None = None):
    array = np.asarray(values, dtype=float)
    if _uses_raw_affinity_scale(dataset_name):
        if np.isscalar(values):
            return float(array)
        return array
    return nm_to_pkd(array)


def prepare_external_regression_frame(
    frame: pd.DataFrame,
    *,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    missing = REQUIRED_EXTERNAL_FRAME_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"External baseline frame is missing columns: {sorted(missing)}")

    out = frame[
        [
            "sample_id",
            "drug_id",
            "target_id",
            "smiles",
            "sequence",
            "label",
            "label_raw",
        ]
    ].copy()
    out["label_raw"] = pd.to_numeric(out["label_raw"], errors="coerce")
    out = out[out["label_raw"].gt(0)].copy()
    out["Drug"] = out["smiles"].astype(str)
    out["Target"] = out["sequence"].astype(str)
    out["Y"] = affinity_to_regression_target(out["label_raw"].to_numpy(), dataset_name=dataset_name)
    out["binary_label"] = out["label"].astype(int)
    out["label_raw_nm"] = pd.to_numeric(out["label_raw"], errors="coerce")
    return out[
        [
            "sample_id",
            "drug_id",
            "target_id",
            "Drug",
            "Target",
            "Y",
            "binary_label",
            "label_raw_nm",
        ]
    ].reset_index(drop=True)


def compute_ranking_metrics(y_true, scores) -> dict[str, float]:
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    try:
        auroc = float(roc_auc_score(labels, scores))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(labels, scores))
    except ValueError:
        auprc = float("nan")
    return {"auroc": auroc, "auprc": auprc}


def subsample_frame(frame: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def save_external_metrics(
    output_dir: str | Path,
    *,
    framework: str,
    model: str,
    dataset: str,
    split_mode: str,
    seed: int,
    metric_split: str,
    metrics: dict,
    run_meta: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "framework": framework,
        "model": model,
        "dataset": dataset,
        "split_mode": split_mode,
        "seed": seed,
        "metric_split": metric_split,
        "metrics": metrics,
        "run_meta": run_meta or {},
    }
    (output_dir / "external_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload

