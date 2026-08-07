from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import split_assay_cold, split_target_cold, split_temporal  # noqa: E402
from mirage_mini.external_baselines import prepare_external_regression_frame, save_external_metrics  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-frame", required=True)
    parser.add_argument("--split-method", choices=["assay_cold", "target_cold", "temporal"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default="outputs/revision_e22_pretrained_frozen_strict_20260804/cache")
    parser.add_argument("--drug-model", default="/home/test/wsk/hf_models/chemberta_77m_mtr")
    parser.add_argument("--protein-model", default="/home/test/wsk/hf_models/esm2_t30_150m_ur50d")
    parser.add_argument("--drug-max-length", type=int, default=512)
    parser.add_argument("--protein-max-length", type=int, default=1024)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--alphas", default="0.03,0.1,0.3,1.0,3.0,10.0,30.0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_from_frame(frame: pd.DataFrame, split_method: str, seed: int) -> dict[str, pd.DataFrame]:
    if split_method == "assay_cold":
        return split_assay_cold(frame, seed=seed)
    if split_method == "target_cold":
        return split_target_cold(frame, seed=seed)
    if split_method == "temporal":
        return split_temporal(frame, seed=seed)
    raise ValueError(split_method)


def prepare_splits(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(args.benchmark_frame)
    raw_splits = split_from_frame(frame, args.split_method, args.seed)
    splits = {
        name: prepare_external_regression_frame(split_frame, dataset_name="CHEMBL_ASSAY")
        for name, split_frame in raw_splits.items()
    }
    limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    for name, limit in limits.items():
        if limit is not None and len(splits[name]) > limit:
            splits[name] = splits[name].sample(n=int(limit), random_state=args.seed).reset_index(drop=True)
    return splits


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(stats.spearmanr(y_true, y_pred)[0])


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    order = np.argsort(y_true)
    y_true = y_true[order]
    y_pred = y_pred[order]
    denom = 0.0
    score = 0.0
    for i in range(1, len(y_true)):
        valid = y_true[i] > y_true[:i]
        if not np.any(valid):
            continue
        diff = y_pred[i] - y_pred[:i][valid]
        denom += float(len(diff))
        score += float(np.sum(diff > 0) + 0.5 * np.sum(diff == 0))
    return float(score / denom) if denom else float("nan")


def ranking_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except ValueError:
        auroc = float("nan")
    try:
        auprc = float(average_precision_score(y_true, scores))
    except ValueError:
        auprc = float("nan")
    return {"auprc": auprc, "auroc": auroc}


def model_cache_stem(model_path: str, max_length: int, prefix: str, values: list[str]) -> str:
    digest = hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest()[:12]
    model_name = Path(model_path).name.replace("/", "_")
    return f"{prefix}_{model_name}_len{max_length}_{digest}"


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (last_hidden * mask).sum(dim=1) / denom


def encode_values(
    *,
    values: list[str],
    model_path: str,
    max_length: int,
    batch_size: int,
    cache_dir: Path,
    prefix: str,
    device: torch.device,
) -> dict[str, np.ndarray]:
    values = sorted({str(v) for v in values if str(v)})
    stem = model_cache_stem(model_path, max_length, prefix, values)
    cache_npz = cache_dir / f"{stem}.npz"
    if cache_npz.exists():
        loaded = np.load(cache_npz, allow_pickle=True)
        keys = [str(x) for x in loaded["keys"].tolist()]
        emb = loaded["embeddings"].astype(np.float32)
        return {key: emb[i] for i, key in enumerate(keys)}

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()
    chunks = []
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(values), batch_size):
            batch_values = values[i : i + batch_size]
            encoded = tokenizer(
                batch_values,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model(**encoded)
            pooled = mean_pool(output.last_hidden_state, encoded["attention_mask"])
            chunks.append(pooled.detach().cpu().to(torch.float32).numpy())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    embeddings = np.vstack(chunks).astype(np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_npz,
        keys=np.asarray(values, dtype=object),
        embeddings=embeddings,
        model_path=str(model_path),
        max_length=int(max_length),
        elapsed_seconds=float(time.perf_counter() - start),
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {key: embeddings[i] for i, key in enumerate(values)}


def featurize(frame: pd.DataFrame, drug_map: dict[str, np.ndarray], protein_map: dict[str, np.ndarray]) -> np.ndarray:
    drug = np.vstack([drug_map[str(x)] for x in frame["Drug"]]).astype(np.float32)
    protein = np.vstack([protein_map[str(x)] for x in frame["Target"]]).astype(np.float32)
    return np.hstack([drug, protein]).astype(np.float32)


def fit_ridge_with_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: list[float],
) -> tuple[Ridge, float, float]:
    best_model = None
    best_alpha = None
    best_rmse = float("inf")
    for alpha in alphas:
        model = Ridge(alpha=float(alpha), random_state=0)
        model.fit(x_train, y_train)
        pred = model.predict(x_val)
        score = rmse(y_val, pred)
        if score < best_rmse:
            best_rmse = score
            best_alpha = float(alpha)
            best_model = model
    if best_model is None or best_alpha is None:
        raise RuntimeError("No ridge model was selected.")
    return best_model, best_alpha, best_rmse


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    splits = prepare_splits(args)
    for name, split in splits.items():
        split.to_csv(output_dir / f"{name}_strict_split.csv", index=False)

    all_drugs = pd.concat([split["Drug"] for split in splits.values()], ignore_index=True).astype(str).tolist()
    all_proteins = pd.concat([split["Target"] for split in splits.values()], ignore_index=True).astype(str).tolist()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    encode_start = time.perf_counter()
    drug_map = encode_values(
        values=all_drugs,
        model_path=args.drug_model,
        max_length=args.drug_max_length,
        batch_size=args.encode_batch_size,
        cache_dir=cache_dir,
        prefix="drug",
        device=device,
    )
    protein_map = encode_values(
        values=all_proteins,
        model_path=args.protein_model,
        max_length=args.protein_max_length,
        batch_size=max(1, args.encode_batch_size // 4),
        cache_dir=cache_dir,
        prefix="protein",
        device=device,
    )
    encode_seconds = time.perf_counter() - encode_start

    x_train = featurize(splits["train"], drug_map, protein_map)
    x_val = featurize(splits["val"], drug_map, protein_map)
    x_test = featurize(splits["test"], drug_map, protein_map)
    y_train = splits["train"]["Y"].to_numpy(dtype=np.float32)
    y_val = splits["val"]["Y"].to_numpy(dtype=np.float32)
    y_test = splits["test"]["Y"].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    fit_start = time.perf_counter()
    model, best_alpha, best_val_rmse = fit_ridge_with_validation(x_train, y_train, x_val, y_val, alphas)
    fit_seconds = time.perf_counter() - fit_start

    inf_start = time.perf_counter()
    test_pred = model.predict(x_test)
    inference_seconds = time.perf_counter() - inf_start
    test_true_binary = splits["test"]["binary_label"].to_numpy(dtype=int)

    test_frame = splits["test"].reset_index(drop=True).copy()
    out = test_frame[["sample_id", "drug_id", "target_id", "Drug", "Target", "binary_label", "label_raw_nm", "Y"]].copy()
    out["pkd_label"] = y_test
    out["pkd_prediction"] = test_pred
    out.to_csv(output_dir / "test_prediction_with_binary.csv", index=False)

    peak_gpu_memory_mb = None
    gpu_name = None
    if torch.cuda.is_available():
        peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        gpu_name = torch.cuda.get_device_name(device)
    parameter_count = int(x_train.shape[1] + 1)
    metrics = {
        **ranking_metrics(test_true_binary, test_pred),
        "rmse": rmse(y_test, test_pred),
        "mae": float(np.mean(np.abs(y_test - test_pred))),
        "pearson": pearson(y_test, test_pred),
        "spearman": spearman(y_test, test_pred),
        "ci": concordance_index(y_test, test_pred),
    }
    run_meta = {
        "strict_benchmark_frame": str(Path(args.benchmark_frame).resolve()),
        "train_size": int(len(splits["train"])),
        "val_size": int(len(splits["val"])),
        "test_size": int(len(splits["test"])),
        "drug_model": args.drug_model,
        "protein_model": args.protein_model,
        "drug_max_length": int(args.drug_max_length),
        "protein_max_length": int(args.protein_max_length),
        "encoding_batch_size": int(args.encode_batch_size),
        "n_unique_drugs": int(len(drug_map)),
        "n_unique_targets": int(len(protein_map)),
        "feature_dim": int(x_train.shape[1]),
        "parameter_count": parameter_count,
        "alphas": alphas,
        "best_alpha": float(best_alpha),
        "best_val_rmse": float(best_val_rmse),
        "encode_seconds": float(encode_seconds),
        "train_seconds": float(fit_seconds),
        "inference_seconds": float(inference_seconds),
        "inference_seconds_per_1000": float(inference_seconds / max(len(splits["test"]), 1) * 1000.0),
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "metric_split_policy": "test_clean_no_row_filtering",
    }
    payload = save_external_metrics(
        output_dir=output_dir,
        framework="pretrained_frozen",
        model="frozen_esm2_chemberta_ridge",
        dataset="CHEMBL_ASSAY",
        split_mode=args.split_method,
        seed=args.seed,
        metric_split="test_clean",
        metrics=metrics,
        run_meta=run_meta,
    )
    (output_dir / "efficiency.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

