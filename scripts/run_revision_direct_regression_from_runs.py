from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.external_baselines import affinity_to_regression_target  # noqa: E402
from mirage_mini.features import MorganFingerprintFeaturizer, MultiModalFeaturizer  # noqa: E402
from mirage_mini.reporting import infer_seed_from_run_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--patterns", nargs="*", default=[])
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--output-dir", default="outputs/revision_e04_direct_regression")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-ci-pairs", type=int, default=200000)
    return parser.parse_args()


def resolve_run_dirs(outputs_root: Path, patterns: list[str], explicit: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(outputs_root.glob(pattern))
    paths.extend(Path(path) for path in explicit)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(paths, key=lambda p: str(p)):
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if (path / "metrics.json").exists() and (path / "train_preview.csv").exists() and (path / "predictions.csv").exists():
            unique.append(path)
    if not unique:
        raise ValueError("No usable run directories matched the requested inputs.")
    return unique


def _subsample(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def _dataset_from_metrics(metrics: dict) -> str:
    return str(metrics.get("dataset") or "UNKNOWN")


def _target_values(df: pd.DataFrame, dataset: str) -> np.ndarray:
    raw = pd.to_numeric(df["label_raw"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(raw)
    if dataset.upper().strip() != "KIBA":
        valid &= raw > 0
    if not np.all(valid):
        raw = raw[valid]
    return np.asarray(affinity_to_regression_target(raw, dataset_name=dataset), dtype=float)


def _usable_regression_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    out["label_raw"] = pd.to_numeric(out["label_raw"], errors="coerce")
    out = out[out["label_raw"].notna()].copy()
    if dataset.upper().strip() != "KIBA":
        out = out[out["label_raw"].gt(0)].copy()
    for col in ["smiles", "sequence", "text"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    return out.reset_index(drop=True)


def _missing_variant(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "text_missing" in out.columns:
        mask = out["text_missing"].astype(str).str.lower().isin(["true", "1", "yes"])
        out.loc[mask, "text"] = ""
    if "sequence_missing" in out.columns:
        mask = out["sequence_missing"].astype(str).str.lower().isin(["true", "1", "yes"])
        out.loc[mask, "sequence"] = ""
    return out


def _fit_current_regressor(train: pd.DataFrame, y_train: np.ndarray) -> tuple[MultiModalFeaturizer, Ridge]:
    featurizer = MultiModalFeaturizer(smiles_features=1024, sequence_features=1024, text_features=512, include_masks=True)
    x_train = featurizer.transform(train).matrix
    model = Ridge(alpha=5.0, random_state=0)
    model.fit(x_train, y_train)
    return featurizer, model


def _predict_current(featurizer: MultiModalFeaturizer, model: Ridge, frame: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(featurizer.transform(frame).matrix), dtype=float)


def _tanimoto_knn_regression(
    train: pd.DataFrame,
    y_train: np.ndarray,
    query: pd.DataFrame,
    *,
    k: int,
    chunk_size: int,
) -> np.ndarray:
    featurizer = MorganFingerprintFeaturizer(n_bits=2048, radius=2)
    train_fp = featurizer.transform(train["smiles"]).astype(np.float32).tocsr()
    query_fp = featurizer.transform(query["smiles"]).astype(np.float32).tocsr()
    train_pop = np.asarray(train_fp.sum(axis=1)).ravel().astype(np.float32)
    query_pop = np.asarray(query_fp.sum(axis=1)).ravel().astype(np.float32)
    train_fp_t = train_fp.T.tocsr()
    fallback = float(np.nanmean(y_train))
    scores: list[float] = []
    for start in range(0, query_fp.shape[0], chunk_size):
        stop = min(start + chunk_size, query_fp.shape[0])
        block = query_fp[start:stop]
        inter = (block @ train_fp_t).toarray().astype(np.float32)
        unions = query_pop[start:stop, None] + train_pop[None, :] - inter
        sims = np.divide(inter, unions, out=np.zeros_like(inter), where=unions > 0)
        for row in sims:
            valid = np.flatnonzero(np.isfinite(row))
            if len(valid) == 0:
                scores.append(fallback)
                continue
            kk = min(k, len(valid))
            top = valid[np.argpartition(-row[valid], kth=kk - 1)[:kk]]
            weights = np.maximum(row[top], 0.0)
            if float(weights.sum()) <= 0:
                scores.append(float(np.mean(y_train[top])))
            else:
                scores.append(float(np.average(y_train[top], weights=weights)))
    return np.asarray(scores, dtype=float)


def _approx_concordance(y_true: np.ndarray, y_pred: np.ndarray, *, max_pairs: int, seed: int) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    dy = y_true[i] - y_true[j]
    dp = y_pred[i] - y_pred[j]
    valid = dy != 0
    if not np.any(valid):
        return float("nan")
    dy = dy[valid]
    dp = dp[valid]
    concordant = np.sum(np.sign(dy) == np.sign(dp))
    tied = np.sum(dp == 0)
    return float((concordant + 0.5 * tied) / len(dy))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, *, max_ci_pairs: int, seed: int) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    if len(y_true) == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "pearson": np.nan, "spearman": np.nan, "ci": np.nan}
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else float("nan")
    spearman = float(stats.spearmanr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "pearson": pearson,
        "spearman": spearman,
        "ci": _approx_concordance(y_true, y_pred, max_pairs=max_ci_pairs, seed=seed),
    }


def _summary_stat(values: pd.Series, stat: str) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "std":
        return float(np.std(arr, ddof=0))
    if stat == "ci95":
        return float(1.96 * np.std(arr, ddof=0) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    raise ValueError(stat)


def analyze_run(
    run_dir: Path,
    *,
    k: int,
    chunk_size: int,
    max_train: int | None,
    max_val: int | None,
    max_test: int | None,
    seed: int,
    max_ci_pairs: int,
) -> list[dict]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
    dataset = _dataset_from_metrics(metrics)
    train = _subsample(_usable_regression_frame(pd.read_csv(run_dir / "train_preview.csv"), dataset), max_train, seed)
    val_path = run_dir / "validation_predictions.csv"
    if val_path.exists():
        val = _subsample(_usable_regression_frame(pd.read_csv(val_path), dataset), max_val, seed + 1)
    else:
        val = train.sample(frac=0.2, random_state=seed + 1).reset_index(drop=True)
    test = _subsample(_usable_regression_frame(pd.read_csv(run_dir / "predictions.csv"), dataset), max_test, seed + 2)
    if len(train) < 5 or len(val) < 5 or len(test) < 5:
        return []

    y_train = _target_values(train, dataset)
    y_val = _target_values(val, dataset)
    y_test = _target_values(test, dataset)
    current_featurizer, current_model = _fit_current_regressor(train, y_train)
    val_current = _predict_current(current_featurizer, current_model, val)
    test_current = _predict_current(current_featurizer, current_model, test)
    missing_test = _missing_variant(test)
    test_current_missing = _predict_current(current_featurizer, current_model, missing_test)

    val_knn = _tanimoto_knn_regression(train, y_train, val, k=k, chunk_size=chunk_size)
    test_knn = _tanimoto_knn_regression(train, y_train, test, k=k, chunk_size=chunk_size)
    test_knn_missing = _tanimoto_knn_regression(train, y_train, missing_test, k=k, chunk_size=chunk_size)

    blend = LinearRegression()
    blend.fit(np.column_stack([val_current, val_knn]), y_val)
    test_fused = blend.predict(np.column_stack([test_current, test_knn]))
    test_fused_missing = blend.predict(np.column_stack([test_current_missing, test_knn_missing]))

    base = {
        "run_dir": run_dir.name,
        "dataset": dataset,
        "split_mode": metrics.get("split_mode"),
        "seed": metrics.get("seed", infer_seed_from_run_dir(run_dir.name)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "k": int(k),
    }
    rows: list[dict] = []
    for metric_split, preds in [
        ("test_clean", {
            "direct_current_ridge": test_current,
            "direct_retrieval_knn": test_knn,
            "direct_two_branch_fusion": test_fused,
        }),
        ("test_missing", {
            "direct_current_ridge": test_current_missing,
            "direct_retrieval_knn": test_knn_missing,
            "direct_two_branch_fusion": test_fused_missing,
        }),
    ]:
        for model, y_pred in preds.items():
            rows.append(
                {
                    **base,
                    "metric_split": metric_split,
                    "model": model,
                    **_metrics(y_test, y_pred, max_ci_pairs=max_ci_pairs, seed=seed),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_root = (REPO_ROOT / args.outputs_root).resolve()
    run_dirs = resolve_run_dirs(outputs_root, args.patterns, args.run_dirs)
    rows: list[dict] = []
    manifest: list[dict] = []
    for run_dir in run_dirs:
        try:
            run_rows = analyze_run(
                run_dir,
                k=args.k,
                chunk_size=args.chunk_size,
                max_train=args.max_train,
                max_val=args.max_val,
                max_test=args.max_test,
                seed=args.seed,
                max_ci_pairs=args.max_ci_pairs,
            )
            rows.extend(run_rows)
            manifest.append({"run_dir": run_dir.name, "status": "ok", "rows": len(run_rows)})
            print(f"[direct-regression] DONE {run_dir.name}: rows={len(run_rows)}", flush=True)
        except Exception as exc:  # noqa: BLE001
            manifest.append({"run_dir": run_dir.name, "status": "failed", "error": repr(exc), "rows": 0})
            print(f"[direct-regression] FAIL {run_dir.name}: {exc!r}", flush=True)

    if not rows:
        raise ValueError("No direct regression rows were generated.")
    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby(["dataset", "split_mode", "metric_split", "model"], dropna=False)
        .agg(
            runs=("run_dir", "nunique"),
            mean_n=("n", "mean"),
            mean_rmse=("rmse", lambda x: _summary_stat(x, "mean")),
            std_rmse=("rmse", lambda x: _summary_stat(x, "std")),
            ci95_rmse=("rmse", lambda x: _summary_stat(x, "ci95")),
            mean_mae=("mae", lambda x: _summary_stat(x, "mean")),
            mean_pearson=("pearson", lambda x: _summary_stat(x, "mean")),
            ci95_pearson=("pearson", lambda x: _summary_stat(x, "ci95")),
            mean_spearman=("spearman", lambda x: _summary_stat(x, "mean")),
            ci95_spearman=("spearman", lambda x: _summary_stat(x, "ci95")),
            mean_ci=("ci", lambda x: _summary_stat(x, "mean")),
            ci95_ci=("ci", lambda x: _summary_stat(x, "ci95")),
        )
        .reset_index()
    )
    per_run.to_csv(output_dir / "direct_regression_per_run.csv", index=False)
    summary.to_csv(output_dir / "direct_regression_summary.csv", index=False)
    pd.DataFrame(manifest).to_csv(output_dir / "run_manifest.csv", index=False)
    report = {"runs_requested": len(run_dirs), "runs_completed": int((pd.DataFrame(manifest)["status"] == "ok").sum()), "rows": len(rows), "output_dir": str(output_dir)}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

