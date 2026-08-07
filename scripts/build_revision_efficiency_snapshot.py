from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.reporting import infer_seed_from_run_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--patterns", nargs="*", default=[])
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--output-dir", default="outputs/revision_e15_efficiency_snapshot")
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
        if (path / "metrics.json").exists() and (path / "predictions.csv").exists():
            unique.append(path)
    if not unique:
        raise ValueError("No usable run directories matched the requested inputs.")
    return unique


def _file_size_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024 * 1024)) if path.exists() else 0.0


def _row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1)
    except OSError:
        return 0


def _write_span_minutes(run_dir: Path) -> float:
    files = [p for p in run_dir.glob("*") if p.is_file()]
    if len(files) < 2:
        return float("nan")
    mtimes = [p.stat().st_mtime for p in files]
    return float((max(mtimes) - min(mtimes)) / 60.0)


def analyze_run(run_dir: Path) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
    pred = pd.read_csv(run_dir / "predictions.csv", nrows=5)
    prob_cols = [col for col in pred.columns if col.endswith("_prob_clean") or col.endswith("_prob_missing")]
    model_names = sorted({col.rsplit("_prob_", 1)[0] for col in prob_cols})
    artifact_mb = sum(_file_size_mb(path) for path in run_dir.glob("*") if path.is_file())
    return {
        "run_dir": run_dir.name,
        "dataset": metrics.get("dataset"),
        "split_mode": metrics.get("split_mode"),
        "seed": metrics.get("seed", infer_seed_from_run_dir(run_dir.name)),
        "n_train": _row_count(run_dir / "train_preview.csv"),
        "n_val": _row_count(run_dir / "validation_predictions.csv"),
        "n_test": _row_count(run_dir / "predictions.csv"),
        "n_probability_columns": int(len(prob_cols)),
        "n_models_detected": int(len(model_names)),
        "artifact_size_mb": artifact_mb,
        "artifact_write_span_min": _write_span_minutes(run_dir),
    }


def _summary_stat(values: pd.Series, stat: str) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "max":
        return float(np.max(arr))
    if stat == "sum":
        return float(np.sum(arr))
    raise ValueError(stat)


def main() -> None:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = resolve_run_dirs(REPO_ROOT / args.outputs_root, args.patterns, args.run_dirs)
    rows = [analyze_run(run_dir) for run_dir in run_dirs]
    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby(["dataset", "split_mode"], dropna=False)
        .agg(
            runs=("run_dir", "nunique"),
            mean_n_train=("n_train", lambda x: _summary_stat(x, "mean")),
            mean_n_test=("n_test", lambda x: _summary_stat(x, "mean")),
            mean_models_detected=("n_models_detected", lambda x: _summary_stat(x, "mean")),
            mean_artifact_size_mb=("artifact_size_mb", lambda x: _summary_stat(x, "mean")),
            max_artifact_size_mb=("artifact_size_mb", lambda x: _summary_stat(x, "max")),
            mean_artifact_write_span_min=("artifact_write_span_min", lambda x: _summary_stat(x, "mean")),
        )
        .reset_index()
    )
    per_run.to_csv(output_dir / "efficiency_snapshot_per_run.csv", index=False)
    summary.to_csv(output_dir / "efficiency_snapshot_summary.csv", index=False)
    report = {"runs": len(per_run), "output_dir": str(output_dir)}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

