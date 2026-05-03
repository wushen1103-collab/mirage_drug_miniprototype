from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, matthews_corrcoef

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import (  # noqa: E402
    inject_missing_modalities,
    prepare_chembl_assay_subset,
    split_assay_cold,
    split_target_cold,
    split_temporal,
)
from mirage_mini.experiment import run_model_suite  # noqa: E402
from mirage_mini.features import CachedTransformerEmbedder  # noqa: E402


MAIN_MODELS = [
    "hybrid_plus_pretrained_smiles_text",
    "hybrid_blend_avg",
    "hybrid_plus_pretrained_smiles_text_retrieval",
    "interaction_gate_tuned",
    "retrieval",
    "mask",
    "no_mask",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-root", default="outputs/submission_hardening/threshold_sensitivity")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[100.0, 300.0, 1000.0, 10000.0])
    parser.add_argument("--splits", nargs="+", default=["assay_cold", "target_cold", "temporal"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--missing-sequence-prob", type=float, default=0.35)
    parser.add_argument("--missing-text-prob", type=float, default=0.35)
    parser.add_argument("--n-neighbors", type=int, default=1)
    parser.add_argument("--pretrained-smiles-model", default="/home/test/wsk/hf_models/chemberta_zinc_base_v1")
    parser.add_argument("--pretrained-text-model", default="/home/test/wsk/hf_models/all_minilm_l6_v2")
    parser.add_argument("--pretrained-text-max-length", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _threshold_slug(threshold: float) -> str:
    raw = f"{threshold:g}".replace(".", "p")
    return f"t{raw}nm"


def _build_embedder(model_name: str, cache_dir: Path, max_length: int, batch_size: int = 64):
    model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower() or "model"
    return CachedTransformerEmbedder(
        model_name=model_name,
        cache_path=cache_dir / "embedding_cache" / f"{model_slug}.pkl",
        batch_size=batch_size,
        max_length=max_length,
    )


def _split_frame(frame: pd.DataFrame, split_mode: str, seed: int) -> dict[str, pd.DataFrame]:
    if split_mode == "assay_cold":
        return split_assay_cold(frame, seed=seed)
    if split_mode == "target_cold":
        return split_target_cold(frame, seed=seed)
    if split_mode == "temporal":
        return split_temporal(frame, seed=seed)
    raise ValueError(f"Unsupported split: {split_mode}")


def _extended_binary(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "brier": float(brier_score_loss(y_true, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }
    return out


def _write_run(
    *,
    output_dir: Path,
    cache_dir: Path,
    threshold_nm: float,
    split_mode: str,
    seed: int,
    args: argparse.Namespace,
    smiles_embedder,
    text_embedder,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists() and (output_dir / "predictions.csv").exists() and not args.overwrite:
        return _summary_from_existing(output_dir)

    bundle = prepare_chembl_assay_subset(
        cache_dir=cache_dir,
        sample_size=args.sample_size,
        seed=seed,
        activity_threshold_nm=threshold_nm,
    )
    splits = _split_frame(bundle.frame, split_mode=split_mode, seed=seed)
    for split_name, frame in splits.items():
        if frame["label"].nunique() < 2:
            raise ValueError(
                f"{split_mode} seed={seed} threshold={threshold_nm:g} has one class in {split_name}: "
                f"prevalence={frame['label'].mean():.4f}, n={len(frame)}"
            )

    stressed_test_df = inject_missing_modalities(
        splits["test"],
        probs={"sequence": args.missing_sequence_prob, "text": args.missing_text_prob},
        seed=seed + 1,
    )
    suite = run_model_suite(
        train_df=splits["train"],
        val_df=splits["val"],
        test_df=splits["test"],
        stressed_test_df=stressed_test_df,
        n_neighbors=args.n_neighbors,
        smiles_embedder=smiles_embedder,
        text_embedder=text_embedder,
        sequence_embedder=None,
        enable_interaction_probe=False,
        retrieval_reference_size=None,
    )

    predictions = suite.pop("predictions")
    validation_predictions = suite.pop("validation_predictions")
    metrics = {
        "dataset": "CHEMBL_ASSAY",
        "threshold_nm": float(threshold_nm),
        "sample_size": int(len(bundle.frame)),
        "target_text_source": bundle.target_text_source,
        "split_mode": split_mode,
        "seed": int(seed),
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "split_prevalence": {name: float(frame["label"].mean()) for name, frame in splits.items()},
        "missing_sequence_prob": args.missing_sequence_prob,
        "missing_text_prob": args.missing_text_prob,
        "n_neighbors": int(args.n_neighbors),
        "models": suite["models"],
    }

    predictions.to_csv(output_dir / "predictions.csv", index=False)
    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    splits["train"].to_csv(output_dir / "train_preview.csv", index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return _summary_from_existing(output_dir)


def _summary_from_existing(output_dir: Path) -> list[dict]:
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8-sig"))
    pred = pd.read_csv(output_dir / "predictions.csv")
    rows: list[dict] = []
    y_true = pred["label"].to_numpy(dtype=int)
    for model in MAIN_MODELS:
        model_payload = metrics.get("models", {}).get(model)
        if not isinstance(model_payload, dict):
            continue
        for metric_split in ["test_clean", "test_missing"]:
            split_payload = model_payload.get(metric_split)
            if not isinstance(split_payload, dict):
                continue
            prob_col = f"{model}_prob_{'clean' if metric_split == 'test_clean' else 'missing'}"
            extra = {}
            if prob_col in pred.columns:
                prob = pd.to_numeric(pred[prob_col], errors="coerce").to_numpy(dtype=float)
                extra = _extended_binary(y_true, prob)
            prevalence = metrics.get("split_prevalence", {}).get("test")
            auprc = split_payload.get("auprc")
            rows.append(
                {
                    "run_dir": output_dir.name,
                    "dataset": metrics.get("dataset"),
                    "threshold_nm": metrics.get("threshold_nm"),
                    "split_mode": metrics.get("split_mode"),
                    "seed": metrics.get("seed"),
                    "metric_split": metric_split,
                    "model": model,
                    "n_test": int(len(pred)),
                    "test_prevalence": prevalence,
                    "auprc": auprc,
                    "auprc_lift": float(auprc - prevalence) if auprc is not None and prevalence is not None else np.nan,
                    "auroc": split_payload.get("auroc"),
                    "ece": split_payload.get("ece"),
                    "risk_at_80_coverage": split_payload.get("risk_at_80_coverage"),
                    **extra,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = (repo_root / args.cache_dir).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    smiles_embedder = _build_embedder(
        model_name=args.pretrained_smiles_model,
        cache_dir=cache_dir,
        max_length=128,
    )
    text_embedder = _build_embedder(
        model_name=args.pretrained_text_model,
        cache_dir=cache_dir,
        max_length=args.pretrained_text_max_length,
    )

    rows: list[dict] = []
    failures: list[dict] = []
    for threshold in args.thresholds:
        for split_mode in args.splits:
            for seed in args.seeds:
                run_dir = output_root / f"chembl_{split_mode}_{_threshold_slug(threshold)}_s{seed}"
                try:
                    print(f"[threshold] START {run_dir.name}", flush=True)
                    rows.extend(
                        _write_run(
                            output_dir=run_dir,
                            cache_dir=cache_dir,
                            threshold_nm=threshold,
                            split_mode=split_mode,
                            seed=seed,
                            args=args,
                            smiles_embedder=smiles_embedder,
                            text_embedder=text_embedder,
                        )
                    )
                    print(f"[threshold] DONE {run_dir.name}", flush=True)
                except Exception as exc:  # keep the grid moving; failed rows are useful evidence too
                    failures.append(
                        {
                            "threshold_nm": threshold,
                            "split_mode": split_mode,
                            "seed": seed,
                            "run_dir": run_dir.name,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                    print(f"[threshold] FAIL {run_dir.name}: {exc}", flush=True)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary.to_csv(output_root / "threshold_sensitivity_long.csv", index=False)
        grouped = (
            summary.groupby(["threshold_nm", "split_mode", "metric_split", "model"], dropna=False)
            .agg(
                runs=("seed", "nunique"),
                n_test=("n_test", "mean"),
                test_prevalence=("test_prevalence", "mean"),
                mean_auprc=("auprc", "mean"),
                std_auprc=("auprc", "std"),
                mean_auprc_lift=("auprc_lift", "mean"),
                mean_auroc=("auroc", "mean"),
                std_auroc=("auroc", "std"),
                mean_ece=("ece", "mean"),
                mean_risk80=("risk_at_80_coverage", "mean"),
                mean_brier=("brier", "mean"),
                mean_balanced_accuracy=("balanced_accuracy", "mean"),
                mean_mcc=("mcc", "mean"),
            )
            .reset_index()
        )
        grouped.to_csv(output_root / "threshold_sensitivity_summary.csv", index=False)
    pd.DataFrame(failures).to_csv(output_root / "threshold_sensitivity_failures.csv", index=False)
    report = {
        "rows": int(len(rows)),
        "failures": int(len(failures)),
        "output_root": str(output_root),
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

