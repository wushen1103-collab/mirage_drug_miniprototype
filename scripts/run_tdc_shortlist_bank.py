from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import inject_missing_modalities, prepare_public_dti_official_splits
from mirage_mini.experiment import run_benchmark_shortlist_suite
from mirage_mini.features import CachedTransformerEmbedder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BindingDB_Kd")
    parser.add_argument(
        "--split-method",
        default="cold_target",
        choices=["random", "cold_drug", "cold_target", "cold_pair"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--activity-threshold-nm", type=float, default=1000.0)
    parser.add_argument("--inactive-threshold-nm", type=float, default=None)
    parser.add_argument("--n-neighbors", type=int, default=8)
    parser.add_argument("--bank-size", type=int, default=1024)
    parser.add_argument("--missing-sequence-prob", type=float, default=0.35)
    parser.add_argument("--missing-text-prob", type=float, default=0.35)
    parser.add_argument("--retrieval-n-jobs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--disable-pretrained-smiles", action="store_true")
    parser.add_argument("--disable-pretrained-text", action="store_true")
    parser.add_argument("--pretrained-smiles-model", default="/home/test/wsk/hf_models/chemberta_zinc_base_v1")
    parser.add_argument("--pretrained-smiles-device", default=None)
    parser.add_argument("--pretrained-smiles-batch-size", type=int, default=64)
    parser.add_argument("--pretrained-smiles-max-length", type=int, default=128)
    parser.add_argument("--pretrained-text-model", default="/home/test/wsk/hf_models/all_minilm_l6_v2")
    parser.add_argument("--pretrained-text-device", default=None)
    parser.add_argument("--pretrained-text-batch-size", type=int, default=64)
    parser.add_argument("--pretrained-text-max-length", type=int, default=128)
    return parser.parse_args()


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return slug or "model"


def _build_embedder(
    enabled: bool,
    model_name: str,
    cache_dir: Path,
    batch_size: int,
    max_length: int,
    device: str | None,
):
    if not enabled:
        return None
    model_slug = _slugify_model_name(model_name)
    return CachedTransformerEmbedder(
        model_name=model_name,
        cache_path=cache_dir / "embedding_cache" / f"{model_slug}.pkl",
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )


def _log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def _rank_models(metrics: dict) -> list[dict]:
    rows = []
    for model_name, model_metrics in metrics["models"].items():
        clean = model_metrics.get("test_clean", {})
        missing = model_metrics.get("test_missing", {})
        rows.append(
            {
                "model": model_name,
                "val_auroc": model_metrics.get("val", {}).get("auroc"),
                "val_auprc": model_metrics.get("val", {}).get("auprc"),
                "test_clean_auroc": clean.get("auroc"),
                "test_clean_auprc": clean.get("auprc"),
                "test_clean_ece": clean.get("ece"),
                "test_missing_auroc": missing.get("auroc"),
                "test_missing_auprc": missing.get("auprc"),
                "test_missing_ece": missing.get("ece"),
            }
        )

    def _sort_key(row: dict) -> tuple[float, float]:
        auroc = row["test_clean_auroc"]
        auprc = row["test_clean_auprc"]
        return (
            float("-inf") if auroc is None else float(auroc),
            float("-inf") if auprc is None else float(auprc),
        )

    rows.sort(key=_sort_key, reverse=True)
    return rows


def main() -> None:
    args = parse_args()
    if args.retrieval_n_jobs is not None:
        os.environ["MIRAGE_RETRIEVAL_N_JOBS"] = str(args.retrieval_n_jobs)

    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_device = args.device
    smiles_device = args.pretrained_smiles_device or shared_device
    text_device = args.pretrained_text_device or shared_device

    smiles_embedder = _build_embedder(
        enabled=not args.disable_pretrained_smiles,
        model_name=args.pretrained_smiles_model,
        cache_dir=cache_dir,
        batch_size=args.pretrained_smiles_batch_size,
        max_length=args.pretrained_smiles_max_length,
        device=smiles_device,
    )
    text_embedder = _build_embedder(
        enabled=not args.disable_pretrained_text,
        model_name=args.pretrained_text_model,
        cache_dir=cache_dir,
        batch_size=args.pretrained_text_batch_size,
        max_length=args.pretrained_text_max_length,
        device=text_device,
    )

    _log("starting split preparation")
    t0 = time.time()
    splits = prepare_public_dti_official_splits(
        dataset_name=args.dataset,
        cache_dir=cache_dir,
        split_method=args.split_method,
        seed=args.seed,
        activity_threshold_nm=args.activity_threshold_nm,
        inactive_threshold_nm=args.inactive_threshold_nm,
    )
    load_seconds = time.time() - t0
    _log(
        "splits ready in "
        f"{load_seconds:.2f}s; sizes train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}"
    )

    stressed_test_df = inject_missing_modalities(
        splits["test"],
        probs={"sequence": args.missing_sequence_prob, "text": args.missing_text_prob},
        seed=args.seed + 1,
    )

    _log("starting shortlist suite")
    t1 = time.time()
    suite = run_benchmark_shortlist_suite(
        train_df=splits["train"],
        val_df=splits["val"],
        test_df=splits["test"],
        stressed_test_df=stressed_test_df,
        n_neighbors=args.n_neighbors,
        smiles_embedder=smiles_embedder,
        text_embedder=text_embedder,
        retrieval_reference_size=args.bank_size,
    )
    suite_seconds = time.time() - t1
    _log(f"shortlist suite finished in {suite_seconds:.2f}s")

    predictions = suite.pop("predictions")
    metrics = suite
    model_summary = _rank_models(metrics)
    meta = {
        "dataset": args.dataset,
        "split_method": args.split_method,
        "seed": args.seed,
        "n_neighbors": args.n_neighbors,
        "retrieval_reference_size": args.bank_size,
        "retrieval_n_jobs": (
            int(os.environ["MIRAGE_RETRIEVAL_N_JOBS"])
            if os.environ.get("MIRAGE_RETRIEVAL_N_JOBS")
            else None
        ),
        "device": shared_device,
        "smiles_device": smiles_device,
        "text_device": text_device,
        "missing_sequence_prob": args.missing_sequence_prob,
        "missing_text_prob": args.missing_text_prob,
        "load_seconds": load_seconds,
        "suite_seconds": suite_seconds,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (output_dir / "model_summary.json").write_text(json.dumps(model_summary, indent=2), encoding="utf-8")
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    _log("artifacts written")
    print(json.dumps({"meta": meta, "top_models": model_summary[:5]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

