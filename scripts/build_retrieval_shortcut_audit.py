from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.features import MorganFingerprintFeaturizer  # noqa: E402
from mirage_mini.metrics import expected_calibration_error, risk_at_coverage  # noqa: E402


AUDIT_CONDITIONS = [
    "original",
    "remove_same_drug",
    "remove_same_target",
    "remove_same_assay",
    "remove_same_pair",
    "remove_high_tanimoto_085",
    "strict_all",
    "label_free_similarity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/submission_hardening/retrieval_shortcut_audit")
    parser.add_argument("--patterns", nargs="*", default=[])
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--high-tanimoto", type=float, default=0.85)
    parser.add_argument("--write-sample-scores", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None)
    return parser.parse_args()


def infer_seed_from_name(name: str):
    match = re.search(r"(?:seed|_s)(\d+)", name)
    return int(match.group(1)) if match else None


def resolve_run_dirs(outputs_root: Path, patterns: list[str], explicit: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(outputs_root.glob(pattern))
    paths.extend(Path(p) for p in explicit)
    unique = []
    seen = set()
    for path in sorted(paths, key=lambda p: str(p)):
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if (path / "metrics.json").exists() and (path / "predictions.csv").exists() and (path / "train_preview.csv").exists():
            unique.append(path)
    return unique


def safe_id(series: pd.Series, default: str = "") -> np.ndarray:
    if series is None:
        return np.array([default])
    return series.fillna(default).astype(str).to_numpy()


def metric_dict(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    mask = np.isfinite(score)
    y_true = y_true[mask]
    score = score[mask]
    if len(y_true) == 0:
        return {
            "auprc": np.nan,
            "auroc": np.nan,
            "ece": np.nan,
            "risk_at_80_coverage": np.nan,
            "brier": np.nan,
            "balanced_accuracy": np.nan,
            "mcc": np.nan,
        }
    try:
        auprc = float(average_precision_score(y_true, score))
    except ValueError:
        auprc = np.nan
    try:
        auroc = float(roc_auc_score(y_true, score))
    except ValueError:
        auroc = np.nan
    pred = (score >= 0.5).astype(int)
    return {
        "auprc": auprc,
        "auroc": auroc,
        "ece": float(expected_calibration_error(y_true, score)),
        "risk_at_80_coverage": float(risk_at_coverage(y_true, score, coverage=0.8)),
        "brier": float(brier_score_loss(y_true, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def _condition_mask(
    condition: str,
    *,
    query_i: int,
    sims: np.ndarray,
    train_drug: np.ndarray,
    train_target: np.ndarray,
    train_assay: np.ndarray,
    query_drug: np.ndarray,
    query_target: np.ndarray,
    query_assay: np.ndarray,
    high_tanimoto: float,
) -> np.ndarray:
    mask = np.ones(len(train_drug), dtype=bool)
    qd = query_drug[query_i]
    qt = query_target[query_i]
    qa = query_assay[query_i]
    if condition in {"remove_same_drug", "strict_all"}:
        mask &= train_drug != qd
    if condition in {"remove_same_target", "strict_all"}:
        mask &= train_target != qt
    if condition in {"remove_same_assay", "strict_all"} and qa:
        mask &= train_assay != qa
    if condition in {"remove_same_pair", "strict_all"}:
        mask &= ~((train_drug == qd) & (train_target == qt))
    if condition in {"remove_high_tanimoto_085", "strict_all"}:
        mask &= sims < high_tanimoto
    return mask


def _topk(values: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
    valid = np.flatnonzero(mask & np.isfinite(values))
    if len(valid) == 0:
        return valid
    k = min(k, len(valid))
    local = np.argpartition(-values[valid], kth=k - 1)[:k]
    idx = valid[local]
    order = np.argsort(-values[idx])
    return idx[order]


def audit_run(run_dir: Path, *, k: int, chunk_size: int, high_tanimoto: float) -> tuple[list[dict], list[dict]]:
    metrics_payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
    train = pd.read_csv(run_dir / "train_preview.csv")
    pred = pd.read_csv(run_dir / "predictions.csv")
    if train.empty or pred.empty:
        return [], []
    if "smiles" not in train.columns or "smiles" not in pred.columns:
        print(f"[audit] SKIP {run_dir.name}: missing smiles column in train/test artifacts", flush=True)
        return [], []

    featurizer = MorganFingerprintFeaturizer(n_bits=2048, radius=2)
    train_fp = featurizer.transform(train["smiles"]).astype(np.float32).tocsr()
    query_fp = featurizer.transform(pred["smiles"]).astype(np.float32).tocsr()
    train_pop = np.asarray(train_fp.sum(axis=1)).ravel().astype(np.float32)
    query_pop = np.asarray(query_fp.sum(axis=1)).ravel().astype(np.float32)

    train_y = train["label"].to_numpy(dtype=int)
    query_y = pred["label"].to_numpy(dtype=int)
    train_drug = safe_id(train.get("drug_id", pd.Series([""] * len(train))))
    train_target = safe_id(train.get("target_id", pd.Series([""] * len(train))))
    train_assay = safe_id(train.get("assay_id", pd.Series([""] * len(train))))
    query_drug = safe_id(pred.get("drug_id", pd.Series([""] * len(pred))))
    query_target = safe_id(pred.get("target_id", pd.Series([""] * len(pred))))
    query_assay = safe_id(pred.get("assay_id", pd.Series([""] * len(pred))))

    sample_rows: list[dict] = []
    run_meta = {
        "run_dir": run_dir.name,
        "dataset": metrics_payload.get("dataset"),
        "split_mode": metrics_payload.get("split_mode"),
        "seed": metrics_payload.get("seed", infer_seed_from_name(run_dir.name)),
        "threshold_nm": metrics_payload.get("threshold_nm", 1000.0),
        "n_train": int(len(train)),
        "n_test": int(len(pred)),
        "test_prevalence": float(np.mean(query_y)),
    }

    score_store = {condition: [] for condition in AUDIT_CONDITIONS}
    neighbor_count_store = {condition: [] for condition in AUDIT_CONDITIONS}
    consistency_store = {condition: [] for condition in AUDIT_CONDITIONS}
    mean_sim_store = {condition: [] for condition in AUDIT_CONDITIONS}
    same_drug_store: list[float] = []
    same_target_store: list[float] = []
    same_assay_store: list[float] = []
    high_sim_store: list[float] = []

    train_fp_t = train_fp.T.tocsr()
    for start in range(0, query_fp.shape[0], chunk_size):
        stop = min(start + chunk_size, query_fp.shape[0])
        block = query_fp[start:stop]
        inter = (block @ train_fp_t).toarray().astype(np.float32)
        unions = query_pop[start:stop, None] + train_pop[None, :] - inter
        sims_block = np.divide(inter, unions, out=np.zeros_like(inter), where=unions > 0)
        for local_i, sims in enumerate(sims_block):
            i = start + local_i
            original_idx = _topk(sims, np.ones(len(train_y), dtype=bool), k)
            if len(original_idx):
                same_drug_store.append(float(np.mean(train_drug[original_idx] == query_drug[i])))
                same_target_store.append(float(np.mean(train_target[original_idx] == query_target[i])))
                same_assay_store.append(float(np.mean((train_assay[original_idx] == query_assay[i]) & (query_assay[i] != ""))))
                high_sim_store.append(float(np.mean(sims[original_idx] >= high_tanimoto)))
            else:
                same_drug_store.append(np.nan)
                same_target_store.append(np.nan)
                same_assay_store.append(np.nan)
                high_sim_store.append(np.nan)

            for condition in AUDIT_CONDITIONS:
                if condition == "label_free_similarity":
                    idx = original_idx
                    score = float(np.mean(sims[idx])) if len(idx) else 0.0
                else:
                    mask = _condition_mask(
                        condition,
                        query_i=i,
                        sims=sims,
                        train_drug=train_drug,
                        train_target=train_target,
                        train_assay=train_assay,
                        query_drug=query_drug,
                        query_target=query_target,
                        query_assay=query_assay,
                        high_tanimoto=high_tanimoto,
                    )
                    idx = _topk(sims, mask, k)
                    score = float(np.mean(train_y[idx])) if len(idx) else float(np.mean(train_y))
                score_store[condition].append(score)
                neighbor_count_store[condition].append(int(len(idx)))
                if len(idx):
                    consistency_store[condition].append(float(np.mean(train_y[idx] == query_y[i])))
                    mean_sim_store[condition].append(float(np.mean(sims[idx])))
                else:
                    consistency_store[condition].append(np.nan)
                    mean_sim_store[condition].append(np.nan)
                sample_rows.append(
                    {
                        **run_meta,
                        "sample_id": pred.iloc[i].get("sample_id", i),
                        "label": int(query_y[i]),
                        "condition": condition,
                        "neighbor_count": int(len(idx)),
                        "score": score,
                        "neighbor_label_consistency": consistency_store[condition][-1],
                        "mean_similarity": mean_sim_store[condition][-1],
                    }
                )

    summary_rows: list[dict] = []
    for condition in AUDIT_CONDITIONS:
        scores = np.asarray(score_store[condition], dtype=float)
        metric_payload = metric_dict(query_y, scores)
        summary_rows.append(
            {
                **run_meta,
                "condition": condition,
                "k": int(k),
                "mean_neighbor_count": float(np.nanmean(neighbor_count_store[condition])),
                "mean_neighbor_label_consistency": float(np.nanmean(consistency_store[condition])),
                "mean_similarity": float(np.nanmean(mean_sim_store[condition])),
                "original_same_drug_frac": float(np.nanmean(same_drug_store)),
                "original_same_target_frac": float(np.nanmean(same_target_store)),
                "original_same_assay_frac": float(np.nanmean(same_assay_store)),
                "original_high_tanimoto_frac": float(np.nanmean(high_sim_store)),
                **metric_payload,
            }
        )

    # Existing branch outputs provide a useful proxy for whether the learned gate moves toward retrieval.
    branch_rows = branch_reliance_summary(pred, run_meta)
    summary_rows.extend(branch_rows)
    return summary_rows, sample_rows


def branch_reliance_summary(pred: pd.DataFrame, run_meta: dict) -> list[dict]:
    rows: list[dict] = []
    candidates = [
        ("test_missing", "hybrid_plus_pretrained_smiles_text_prob_missing", "hybrid_plus_pretrained_smiles_text_retrieval_prob_missing", "interaction_gate_tuned_prob_missing"),
        ("test_clean", "hybrid_plus_pretrained_smiles_text_prob_clean", "hybrid_plus_pretrained_smiles_text_retrieval_prob_clean", "interaction_gate_tuned_prob_clean"),
    ]
    for metric_split, current_col, retrieval_col, fused_col in candidates:
        if not {current_col, retrieval_col, fused_col}.issubset(pred.columns):
            continue
        current = pd.to_numeric(pred[current_col], errors="coerce").to_numpy(dtype=float)
        retrieval = pd.to_numeric(pred[retrieval_col], errors="coerce").to_numpy(dtype=float)
        fused = pd.to_numeric(pred[fused_col], errors="coerce").to_numpy(dtype=float)
        conflict = np.abs(current - retrieval)
        reliance = np.clip(np.abs(fused - current) / (np.abs(retrieval - current) + 1e-8), 0.0, 1.0)
        q = pd.qcut(pd.Series(conflict), q=4, labels=False, duplicates="drop")
        for quartile, idx in q.groupby(q).groups.items():
            idx = np.asarray(list(idx), dtype=int)
            rows.append(
                {
                    **run_meta,
                    "condition": f"branch_reliance_q{int(quartile) + 1}_{metric_split}",
                    "k": np.nan,
                    "mean_neighbor_count": np.nan,
                    "mean_neighbor_label_consistency": np.nan,
                    "mean_similarity": np.nan,
                    "original_same_drug_frac": np.nan,
                    "original_same_target_frac": np.nan,
                    "original_same_assay_frac": np.nan,
                    "original_high_tanimoto_frac": np.nan,
                    "auprc": np.nan,
                    "auroc": np.nan,
                    "ece": np.nan,
                    "risk_at_80_coverage": np.nan,
                    "brier": np.nan,
                    "balanced_accuracy": np.nan,
                    "mcc": np.nan,
                    "mean_conflict": float(np.nanmean(conflict[idx])),
                    "mean_reliance_proxy": float(np.nanmean(reliance[idx])),
                    "n_quartile": int(len(idx)),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = (repo_root / args.outputs_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = resolve_run_dirs(outputs_root, args.patterns, args.run_dirs)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]
    if not run_dirs:
        raise ValueError("No run directories matched the requested patterns.")

    all_summary: list[dict] = []
    all_samples: list[dict] = []
    manifest: list[dict] = []
    for run_dir in run_dirs:
        print(f"[audit] START {run_dir.name}", flush=True)
        summary, samples = audit_run(
            run_dir,
            k=args.k,
            chunk_size=args.chunk_size,
            high_tanimoto=args.high_tanimoto,
        )
        all_summary.extend(summary)
        if args.write_sample_scores:
            all_samples.extend(samples)
        manifest.append({"run_dir": run_dir.name, "summary_rows": len(summary), "sample_rows": len(samples)})
        print(f"[audit] DONE {run_dir.name}: summary_rows={len(summary)}", flush=True)

    summary_df = pd.DataFrame(all_summary)
    summary_df.to_csv(output_dir / "retrieval_shortcut_audit_summary.csv", index=False)
    if not summary_df.empty:
        metric_rows = summary_df[summary_df["condition"].isin(AUDIT_CONDITIONS)].copy()
        grouped = (
            metric_rows.groupby(["dataset", "split_mode", "condition"], dropna=False)
            .agg(
                runs=("run_dir", "nunique"),
                mean_test_prevalence=("test_prevalence", "mean"),
                mean_neighbor_count=("mean_neighbor_count", "mean"),
                mean_neighbor_label_consistency=("mean_neighbor_label_consistency", "mean"),
                mean_similarity=("mean_similarity", "mean"),
                original_same_drug_frac=("original_same_drug_frac", "mean"),
                original_same_target_frac=("original_same_target_frac", "mean"),
                original_same_assay_frac=("original_same_assay_frac", "mean"),
                original_high_tanimoto_frac=("original_high_tanimoto_frac", "mean"),
                mean_auprc=("auprc", "mean"),
                mean_auroc=("auroc", "mean"),
                mean_ece=("ece", "mean"),
                mean_risk80=("risk_at_80_coverage", "mean"),
                mean_brier=("brier", "mean"),
                mean_balanced_accuracy=("balanced_accuracy", "mean"),
                mean_mcc=("mcc", "mean"),
            )
            .reset_index()
        )
        grouped.to_csv(output_dir / "retrieval_shortcut_audit_grouped.csv", index=False)
        reliance = summary_df[summary_df["condition"].str.startswith("branch_reliance", na=False)].copy()
        reliance.to_csv(output_dir / "branch_reliance_proxy.csv", index=False)

    if args.write_sample_scores:
        pd.DataFrame(all_samples).to_csv(output_dir / "retrieval_shortcut_sample_scores.csv", index=False)
    pd.DataFrame(manifest).to_csv(output_dir / "run_manifest.csv", index=False)
    report = {
        "runs_analyzed": int(len(run_dirs)),
        "summary_rows": int(len(all_summary)),
        "sample_rows_written": int(len(all_samples)) if args.write_sample_scores else 0,
        "output_dir": str(output_dir),
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

