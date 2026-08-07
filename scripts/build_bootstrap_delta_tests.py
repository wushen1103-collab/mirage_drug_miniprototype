from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.metrics import expected_calibration_error, risk_at_coverage  # noqa: E402


FIXED_MODEL = "mirage_full"
BASELINE_MODELS = [
    "mirage_w_o_gate",
    "mirage_w_o_probe",
    "mirage_w_o_anchor",
    "hybrid_blend_avg",
    "mask",
    "retrieval",
    "historical_retrieval_evidence",
    "hybrid_plus_pretrained_smiles_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/submission_hardening/bootstrap_delta_tests")
    parser.add_argument("--patterns", nargs="*", default=[])
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
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
        if (path / "metrics.json").exists() and (path / "predictions.csv").exists():
            unique.append(path)
    return unique


def metric_values(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    out = {}
    try:
        out["auprc"] = float(average_precision_score(y, prob))
    except ValueError:
        out["auprc"] = np.nan
    try:
        out["auroc"] = float(roc_auc_score(y, prob))
    except ValueError:
        out["auroc"] = np.nan
    out["ece"] = float(expected_calibration_error(y, prob))
    out["risk80"] = float(risk_at_coverage(y, prob, coverage=0.8))
    out["brier"] = float(brier_score_loss(y, prob))
    return out


def bootstrap_delta(y: np.ndarray, fixed: np.ndarray, baseline: np.ndarray, n_bootstrap: int, rng: np.random.Generator):
    n = len(y)
    full_fixed = metric_values(y, fixed)
    full_base = metric_values(y, baseline)
    rows = []
    boot = {metric: [] for metric in ["auprc", "auroc", "ece", "risk80", "brier"]}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        mf = metric_values(y[idx], fixed[idx])
        mb = metric_values(y[idx], baseline[idx])
        for metric in boot:
            boot[metric].append(mf[metric] - mb[metric])
    for metric, values in boot.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        rows.append(
            {
                "metric": metric,
                "fixed_value": full_fixed[metric],
                "baseline_value": full_base[metric],
                "delta_fixed_minus_baseline": full_fixed[metric] - full_base[metric],
                "ci95_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci95_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                "p_delta_le_0": float(np.mean(values <= 0)) if len(values) else np.nan,
                "p_delta_ge_0": float(np.mean(values >= 0)) if len(values) else np.nan,
                "bootstrap_n": int(len(values)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = (repo_root / args.outputs_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    rows: list[dict] = []
    manifest: list[dict] = []
    for run_dir in resolve_run_dirs(outputs_root, args.patterns, args.run_dirs):
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
        pred = pd.read_csv(run_dir / "predictions.csv")
        y = pred["label"].to_numpy(dtype=int)
        for metric_split, suffix in [("test_clean", "clean"), ("test_missing", "missing")]:
            fixed_col = f"{FIXED_MODEL}_prob_{suffix}"
            if fixed_col not in pred.columns:
                continue
            fixed = pd.to_numeric(pred[fixed_col], errors="coerce").to_numpy(dtype=float)
            for baseline in BASELINE_MODELS:
                base_col = f"{baseline}_prob_{suffix}"
                if base_col not in pred.columns:
                    continue
                base = pd.to_numeric(pred[base_col], errors="coerce").to_numpy(dtype=float)
                valid = np.isfinite(fixed) & np.isfinite(base)
                if valid.sum() < 10 or len(np.unique(y[valid])) < 2:
                    continue
                for row in bootstrap_delta(y[valid], fixed[valid], base[valid], args.n_bootstrap, rng):
                    rows.append(
                        {
                            "run_dir": run_dir.name,
                            "dataset": metrics.get("dataset"),
                            "split_mode": metrics.get("split_mode"),
                            "threshold_nm": metrics.get("threshold_nm", 1000.0),
                            "seed": metrics.get("seed", infer_seed_from_name(run_dir.name)),
                            "metric_split": metric_split,
                            "fixed_model": FIXED_MODEL,
                            "baseline_model": baseline,
                            "n": int(valid.sum()),
                            **row,
                        }
                    )
        manifest.append({"run_dir": run_dir.name, "rows": len(rows)})

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "bootstrap_delta_long.csv", index=False)
    if not out.empty:
        summary = (
            out.groupby(["dataset", "split_mode", "threshold_nm", "metric_split", "baseline_model", "metric"], dropna=False)
            .agg(
                runs=("run_dir", "nunique"),
                mean_delta=("delta_fixed_minus_baseline", "mean"),
                mean_ci95_low=("ci95_low", "mean"),
                mean_ci95_high=("ci95_high", "mean"),
                mean_p_delta_le_0=("p_delta_le_0", "mean"),
                mean_p_delta_ge_0=("p_delta_ge_0", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(output_dir / "bootstrap_delta_summary.csv", index=False)
    pd.DataFrame(manifest).to_csv(output_dir / "run_manifest.csv", index=False)
    report = {"rows": int(len(rows)), "output_dir": str(output_dir)}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

