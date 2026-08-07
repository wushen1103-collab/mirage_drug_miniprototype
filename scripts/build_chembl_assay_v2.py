from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.data import (  # noqa: E402
    _fetch_chembl_assay_metadata,
    _fetch_chembl_target_metadata,
    _fetch_uniprot_metadata,
    split_assay_cold,
    split_random,
    split_target_cold,
    split_temporal,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a larger, assay-aware ChEMBL distribution-shift benchmark."
    )
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--output-dir", default="outputs/chembl_assay_revision_20260729")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--standard-types", nargs="+", default=["IC50", "Ki", "Kd", "EC50"])
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--fetch-mode", choices=["full", "capped"], default="full")
    parser.add_argument("--pages-per-year-type", type=int, default=2)
    parser.add_argument("--max-pages-per-year-type", type=int, default=0)
    parser.add_argument("--allow-incomplete-fetch", action="store_true")
    parser.add_argument(
        "--activity-cache-namespace",
        default="chembl_assay_revision_activity_pages",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--raw-activity-candidates",
        default="",
        help=(
            "Optional complete raw_activity_candidates.csv archive to reuse. When supplied, "
            "the adjacent fetch_summary.csv is reused for the fetch audit instead of querying "
            "the ChEMBL activity endpoint again."
        ),
    )
    parser.add_argument("--target-organism", default="Homo sapiens")
    parser.add_argument("--allowed-assay-types", nargs="+", default=["B", "F"])
    parser.add_argument("--min-assay-records", type=int, default=8)
    parser.add_argument(
        "--label-mode",
        choices=["absolute_pchembl", "assay_quantile"],
        default="absolute_pchembl",
        help="Use an absolute potency threshold for the primary task or the legacy within-assay ranking labels.",
    )
    parser.add_argument(
        "--activity-threshold-nm",
        type=float,
        default=1000.0,
        help="Absolute active threshold in nM when --label-mode=absolute_pchembl.",
    )
    parser.add_argument("--low-quantile", type=float, default=0.40)
    parser.add_argument("--high-quantile", type=float, default=0.60)
    parser.add_argument("--min-pchembl-gap", type=float, default=0.20)
    parser.add_argument("--max-per-assay-class", type=int, default=40)
    parser.add_argument(
        "--max-assays",
        type=int,
        default=0,
        help="Deterministically retain at most this many assays after metadata filtering; 0 keeps all assays.",
    )
    parser.add_argument(
        "--max-records-per-assay",
        type=int,
        default=0,
        help=(
            "For absolute-threshold tasks, retain at most this many pChEMBL-quantile-spaced "
            "records per selected assay; 0 keeps all records."
        ),
    )
    parser.add_argument("--min-confidence-score", type=int, default=8)
    parser.add_argument(
        "--metadata-mode",
        choices=["activity_fields", "api_verified"],
        default="activity_fields",
        help=(
            "Use assay text/type/target identifiers already returned by the full ChEMBL "
            "activity endpoint, or additionally re-query every individual assay record. "
            "The activity-fields mode is the reproducible primary protocol; api_verified "
            "is retained for an optional metadata audit."
        ),
    )
    parser.add_argument("--overwrite-fetch-cache", action="store_true")
    return parser.parse_args()


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def _request_json(url: str, timeout_sec: int = 60, max_retries: int = 4) -> dict:
    headers = {"User-Agent": "mirage-chembl-assay-v2/0.1"}
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout_sec, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}") from last_error


def _activity_cache_path(
    cache_dir: Path,
    cache_namespace: str,
    standard_type: str,
    year: int,
    page_size: int,
    offset: int,
) -> Path:
    safe_type = standard_type.replace("/", "_")
    return (
        cache_dir
        / cache_namespace
        / f"activity_{safe_type}_y{year}_limit{page_size}_offset{offset}.json"
    )


def _activity_url(
    standard_type: str,
    year: int,
    page_size: int,
    offset: int,
    *,
    target_organism: str,
    allowed_assay_types: Iterable[str],
) -> str:
    params = {
        "limit": page_size,
        "offset": offset,
        "standard_units": "nM",
        "standard_relation": "=",
        "standard_type": standard_type,
        "document_year": year,
        "pchembl_value__isnull": "false",
        "potential_duplicate": "false",
        "data_validity_comment__isnull": "true",
        "canonical_smiles__isnull": "false",
        "order_by": "activity_id",
    }
    if target_organism:
        params["target_organism"] = target_organism
    allowed_assay_types = tuple(allowed_assay_types)
    if allowed_assay_types:
        params["assay_type__in"] = ",".join(allowed_assay_types)
    return "https://www.ebi.ac.uk/chembl/api/data/activity.json?" + urlencode(params)


def _fetch_activity_task(
    *,
    cache_dir: Path,
    standard_type: str,
    year: int,
    page_size: int,
    offset: int,
    cache_namespace: str,
    target_organism: str,
    allowed_assay_types: Iterable[str],
    overwrite: bool,
) -> dict:
    cache_path = _activity_cache_path(
        cache_dir,
        cache_namespace,
        standard_type,
        year,
        page_size,
        offset,
    )
    if cache_path.exists() and not overwrite:
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    payload = _request_json(
        _activity_url(
            standard_type,
            year,
            page_size,
            offset,
            target_organism=target_organism,
            allowed_assay_types=allowed_assay_types,
        )
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _append_activity_rows(rows: list[dict], activities: Iterable[dict], args: argparse.Namespace) -> None:
    for obj in activities:
        try:
            standard_value_nm = float(obj.get("standard_value"))
            pchembl_value = float(obj.get("pchembl_value"))
        except Exception:
            continue
        if not np.isfinite(standard_value_nm) or not np.isfinite(pchembl_value):
            continue
        if standard_value_nm <= 0:
            continue
        if obj.get("potential_duplicate"):
            continue
        if str(obj.get("data_validity_comment") or "").strip():
            continue
        if args.target_organism and obj.get("target_organism") != args.target_organism:
            continue
        assay_type = str(obj.get("assay_type") or "").strip()
        if args.allowed_assay_types and assay_type not in set(args.allowed_assay_types):
            continue
        smiles = str(obj.get("canonical_smiles") or "").strip()
        assay_text = str(obj.get("assay_description") or "").strip()
        if not smiles or not assay_text:
            continue
        rows.append(
            {
                "activity_id": obj.get("activity_id"),
                "drug_id": obj.get("molecule_chembl_id"),
                "target_id": obj.get("target_chembl_id"),
                "assay_id": obj.get("assay_chembl_id"),
                "standard_type": obj.get("standard_type"),
                "assay_type": assay_type,
                "document_year": obj.get("document_year"),
                "smiles": smiles,
                "assay_text": assay_text,
                "target_pref_name": obj.get("target_pref_name"),
                "target_organism": obj.get("target_organism"),
                "label_raw": standard_value_nm,
                "pchembl_value": pchembl_value,
            }
        )


def collect_activity_rows(args: argparse.Namespace, cache_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    combinations = [
        (standard_type, year)
        for year in range(args.start_year, args.end_year + 1)
        for standard_type in args.standard_types
    ]
    rows: list[dict] = []
    fetch_rows: list[dict] = []
    totals: dict[tuple[str, int], int] = {}

    def collect_tasks(tasks: Iterable[tuple[str, int, int]]) -> None:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    _fetch_activity_task,
                    cache_dir=cache_dir,
                    standard_type=standard_type,
                    year=year,
                    page_size=args.page_size,
                    offset=offset,
                    cache_namespace=args.activity_cache_namespace,
                    target_organism=args.target_organism,
                    allowed_assay_types=args.allowed_assay_types,
                    overwrite=args.overwrite_fetch_cache,
                ): (standard_type, year, offset)
                for standard_type, year, offset in tasks
            }
            for future in as_completed(futures):
                standard_type, year, offset = futures[future]
                try:
                    payload = future.result()
                    error = None
                except Exception as exc:
                    payload = None
                    error = repr(exc)
                activities = [] if not isinstance(payload, dict) else payload.get("activities", [])
                page_meta = {} if not isinstance(payload, dict) else payload.get("page_meta", {})
                total_count = int(page_meta.get("total_count") or 0)
                totals[(standard_type, year)] = max(totals.get((standard_type, year), 0), total_count)
                fetch_rows.append(
                    {
                        "standard_type": standard_type,
                        "year": year,
                        "offset": offset,
                        "requested_limit": args.page_size,
                        "page_count": int(len(activities)),
                        "total_count": total_count,
                        "error": error,
                    }
                )
                _append_activity_rows(rows, activities, args)

    collect_tasks((standard_type, year, 0) for standard_type, year in combinations)

    remaining_tasks: list[tuple[str, int, int]] = []
    for standard_type, year in combinations:
        total_count = totals.get((standard_type, year), 0)
        total_pages = int(math.ceil(total_count / args.page_size)) if total_count else 1
        if args.fetch_mode == "capped":
            pages_to_fetch = min(total_pages, max(1, args.pages_per_year_type))
        else:
            pages_to_fetch = total_pages
        if args.max_pages_per_year_type > 0:
            pages_to_fetch = min(pages_to_fetch, args.max_pages_per_year_type)
        remaining_tasks.extend(
            (standard_type, year, page_idx * args.page_size)
            for page_idx in range(1, pages_to_fetch)
        )
    collect_tasks(remaining_tasks)

    fetch = pd.DataFrame(fetch_rows).sort_values(["year", "standard_type", "offset"]).reset_index(drop=True)
    raw = pd.DataFrame(rows)
    return raw, fetch


def _collapse_replicates(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    raw = raw.dropna(subset=["drug_id", "target_id", "assay_id", "pchembl_value", "label_raw"]).copy()
    raw["document_year"] = pd.to_numeric(raw["document_year"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["document_year"]).copy()
    sort_cols = ["drug_id", "target_id", "assay_id", "standard_type", "document_year"]
    grouped = (
        raw.sort_values(sort_cols, kind="mergesort")
        .groupby(["drug_id", "target_id", "assay_id", "standard_type"], as_index=False)
        .agg(
            {
                "activity_id": "first",
                "assay_type": "first",
                "document_year": "median",
                "smiles": "first",
                "assay_text": "first",
                "target_pref_name": "first",
                "target_organism": "first",
                "label_raw": "median",
                "pchembl_value": "median",
            }
        )
    )
    grouped["document_year"] = grouped["document_year"].round().astype(int)
    return grouped.reset_index(drop=True)


def _filter_by_group_support(frame: pd.DataFrame, group_column: str, min_count: int) -> pd.DataFrame:
    counts = frame[group_column].value_counts()
    keep = counts[counts >= min_count].index
    return frame[frame[group_column].isin(keep)].copy()


def _assay_quantile_label(
    frame: pd.DataFrame,
    *,
    low_quantile: float,
    high_quantile: float,
    min_records: int,
    min_gap: float,
    max_per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_parts: list[pd.DataFrame] = []
    policy_rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for assay_id, group in frame.groupby("assay_id", sort=True):
        group = group.dropna(subset=["pchembl_value"]).copy()
        if len(group) < min_records:
            continue
        low = float(group["pchembl_value"].quantile(low_quantile))
        high = float(group["pchembl_value"].quantile(high_quantile))
        gap = high - low
        if not np.isfinite(low) or not np.isfinite(high) or gap < min_gap:
            continue
        working = group[(group["pchembl_value"] <= low) | (group["pchembl_value"] >= high)].copy()
        if working.empty:
            continue
        working["label"] = (working["pchembl_value"] >= high).astype(int)
        if working["label"].nunique() < 2:
            continue
        if max_per_class > 0:
            sampled = []
            for label, label_group in working.groupby("label"):
                if len(label_group) > max_per_class:
                    sampled.append(label_group.sample(n=max_per_class, random_state=int(rng.integers(1, 2**31 - 1))))
                else:
                    sampled.append(label_group)
            working = pd.concat(sampled, ignore_index=True)
        working["assay_low_pchembl"] = low
        working["assay_high_pchembl"] = high
        working["assay_threshold_policy"] = f"within-assay q{low_quantile:.2f}/q{high_quantile:.2f}"
        policy_rows.append(
            {
                "assay_id": assay_id,
                "records_before_labeling": int(len(group)),
                "records_after_labeling": int(len(working)),
                "positive_rate_after_labeling": float(working["label"].mean()),
                "low_pchembl": low,
                "high_pchembl": high,
                "pchembl_gap": gap,
                "min_year": int(group["document_year"].min()),
                "max_year": int(group["document_year"].max()),
                "unique_drugs": int(group["drug_id"].nunique()),
                "unique_targets": int(group["target_id"].nunique()),
            }
        )
        labeled_parts.append(working)

    if not labeled_parts:
        return pd.DataFrame(), pd.DataFrame(policy_rows)
    labeled = pd.concat(labeled_parts, ignore_index=True)
    policy = pd.DataFrame(policy_rows).sort_values(["records_after_labeling", "assay_id"], ascending=[False, True])
    return labeled, policy.reset_index(drop=True)


def _absolute_pchembl_label(
    frame: pd.DataFrame,
    *,
    activity_threshold_nm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create prospective labels without using the distribution of a held-out assay."""
    if activity_threshold_nm <= 0:
        raise ValueError("activity_threshold_nm must be positive")
    active_threshold_pchembl = 9.0 - math.log10(float(activity_threshold_nm))
    labeled = frame.dropna(subset=["pchembl_value"]).copy()
    labeled["label"] = (pd.to_numeric(labeled["pchembl_value"], errors="coerce") >= active_threshold_pchembl).astype(int)
    labeled["assay_low_pchembl"] = np.nan
    labeled["assay_high_pchembl"] = np.nan
    labeled["assay_threshold_policy"] = (
        f"absolute pChEMBL >= {active_threshold_pchembl:.4f} "
        f"(<= {float(activity_threshold_nm):g} nM)"
    )
    policy = (
        labeled.groupby("assay_id", as_index=False)
        .agg(
            records_before_labeling=("assay_id", "size"),
            records_after_labeling=("assay_id", "size"),
            positive_rate_after_labeling=("label", "mean"),
            low_pchembl=("pchembl_value", "min"),
            high_pchembl=("pchembl_value", "max"),
            pchembl_gap=("pchembl_value", lambda values: float(np.max(values) - np.min(values))),
            min_year=("document_year", "min"),
            max_year=("document_year", "max"),
            unique_drugs=("drug_id", "nunique"),
            unique_targets=("target_id", "nunique"),
        )
        .sort_values(["records_after_labeling", "assay_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    return labeled.reset_index(drop=True), policy


def _describe_frame(frame: pd.DataFrame, split_mode: str | None = None, split_name: str | None = None) -> dict:
    pchembl = pd.to_numeric(frame["pchembl_value"], errors="coerce")
    label = pd.to_numeric(frame["label"], errors="coerce")
    years = pd.to_numeric(frame["document_year"], errors="coerce")
    return {
        "split_mode": split_mode,
        "split_name": split_name,
        "rows": int(len(frame)),
        "positive_rate": float(label.mean()) if len(frame) else math.nan,
        "unique_drugs": int(frame["drug_id"].nunique()) if "drug_id" in frame else 0,
        "unique_targets": int(frame["target_id"].nunique()) if "target_id" in frame else 0,
        "unique_assays": int(frame["assay_id"].nunique()) if "assay_id" in frame else 0,
        "min_year": int(years.min()) if years.notna().any() else None,
        "max_year": int(years.max()) if years.notna().any() else None,
        "mean_pchembl": float(pchembl.mean()),
        "median_pchembl": float(pchembl.median()),
        "std_pchembl": float(pchembl.std()),
        "mean_nm": float(pd.to_numeric(frame["label_raw"], errors="coerce").mean()),
        "median_nm": float(pd.to_numeric(frame["label_raw"], errors="coerce").median()),
        "missing_text_rate": float(frame["text"].fillna("").eq("").mean()) if "text" in frame else 0.0,
        "missing_sequence_rate": float(frame["sequence"].fillna("").eq("").mean()) if "sequence" in frame else 0.0,
    }


def _overlap_rows(split_mode: str, frames: dict[str, pd.DataFrame]) -> list[dict]:
    rows: list[dict] = []
    for entity in ["drug_id", "target_id", "assay_id", "drug_target_pair"]:
        if entity == "drug_target_pair":
            values = {
                split: set((df["drug_id"].astype(str) + "::" + df["target_id"].astype(str)).dropna())
                for split, df in frames.items()
            }
        elif entity in frames["train"].columns:
            values = {split: set(df[entity].dropna().astype(str)) for split, df in frames.items()}
        else:
            continue
        for left_name, right_name in [("train", "val"), ("train", "test"), ("val", "test")]:
            left = values[left_name]
            right = values[right_name]
            union = left | right
            overlap = left & right
            rows.append(
                {
                    "split_mode": split_mode,
                    "entity": entity,
                    "pair": f"{left_name}_vs_{right_name}",
                    "left_unique": int(len(left)),
                    "right_unique": int(len(right)),
                    "overlap_count": int(len(overlap)),
                    "jaccard": float(len(overlap) / len(union)) if union else math.nan,
                }
            )
    return rows


def _split_builders(seed: int):
    return {
        "random": lambda frame: split_random(frame, seed=seed),
        "target_cold": lambda frame: split_target_cold(frame, seed=seed),
        "assay_cold": lambda frame: split_assay_cold(frame, seed=seed),
        "temporal": lambda frame: split_temporal(frame, seed=seed),
    }


def _allocation_quotas(counts: pd.Series, limit: int) -> pd.Series:
    """Allocate a fixed assay budget proportionally while preserving non-empty strata."""
    counts = counts.sort_index()
    if limit <= 0 or limit >= int(counts.sum()):
        return counts.astype(int)
    raw = counts.astype(float) / float(counts.sum()) * limit
    quotas = np.floor(raw).astype(int)
    if limit >= len(quotas):
        quotas = quotas.clip(lower=1)
    quotas = np.minimum(quotas, counts.astype(int))
    remaining = int(limit - quotas.sum())
    for stratum in (raw - quotas).sort_values(ascending=False, kind="mergesort").index:
        if remaining <= 0:
            break
        if quotas.loc[stratum] < counts.loc[stratum]:
            quotas.loc[stratum] += 1
            remaining -= 1
    return quotas.astype(int)


def _assay_balanced_regression_subset(
    frame: pd.DataFrame,
    *,
    max_assays: int,
    max_records_per_assay: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a bounded, audit-ready assay-balanced regression cohort.

    Assays are selected by their first observed year and dominant standard type. Within each
    selected assay, evenly spaced pChEMBL ranks preserve the continuous affinity range without
    deriving the selection from a binary activity label.
    """
    if frame.empty:
        return frame.copy(), pd.DataFrame()

    working = frame.copy()
    working["document_year"] = pd.to_numeric(working["document_year"], errors="coerce")
    working["pchembl_value"] = pd.to_numeric(working["pchembl_value"], errors="coerce")
    working = working.dropna(subset=["assay_id", "document_year", "pchembl_value"]).copy()
    assay_meta = (
        working.sort_values(["assay_id", "document_year", "standard_type"], kind="mergesort")
        .groupby("assay_id", as_index=False)
        .agg(
            candidate_rows=("assay_id", "size"),
            first_year=("document_year", "min"),
            last_year=("document_year", "max"),
            dominant_standard_type=("standard_type", "first"),
            min_pchembl=("pchembl_value", "min"),
            median_pchembl=("pchembl_value", "median"),
            max_pchembl=("pchembl_value", "max"),
        )
        .sort_values("assay_id", kind="mergesort")
        .reset_index(drop=True)
    )
    assay_meta["year_stratum"] = pd.cut(
        assay_meta["first_year"],
        bins=[-np.inf, 2014, 2019, np.inf],
        labels=["2010-2014", "2015-2019", "2020-2024"],
    ).astype(str)
    assay_meta["selection_stratum"] = (
        assay_meta["year_stratum"] + "|" + assay_meta["dominant_standard_type"].astype(str)
    )
    assay_meta["selected"] = True
    if max_assays > 0 and len(assay_meta) > max_assays:
        counts = assay_meta.groupby("selection_stratum")["assay_id"].size()
        quotas = _allocation_quotas(counts, max_assays)
        rng = np.random.default_rng(seed)
        selected_ids: list[str] = []
        for stratum, quota in quotas.items():
            candidates = assay_meta.loc[
                assay_meta["selection_stratum"] == stratum, "assay_id"
            ].to_numpy()
            if quota >= len(candidates):
                selected_ids.extend(candidates.tolist())
            else:
                selected_ids.extend(rng.choice(candidates, size=int(quota), replace=False).tolist())
        selected_set = set(selected_ids)
        assay_meta["selected"] = assay_meta["assay_id"].isin(selected_set)

    selected_meta = assay_meta[assay_meta["selected"]].copy()
    selected = working[working["assay_id"].isin(set(selected_meta["assay_id"]))].copy()
    if max_records_per_assay > 0:
        retained_rows: list[pd.DataFrame] = []
        for _, group in selected.groupby("assay_id", sort=True):
            group = group.sort_values(["pchembl_value", "activity_id"], kind="mergesort")
            if len(group) > max_records_per_assay:
                positions = np.linspace(0, len(group) - 1, max_records_per_assay).round().astype(int)
                group = group.iloc[np.unique(positions)]
            retained_rows.append(group)
        selected = pd.concat(retained_rows, ignore_index=True) if retained_rows else selected.iloc[0:0].copy()

    retained = selected.groupby("assay_id").size().rename("retained_rows")
    assay_meta = assay_meta.merge(retained, left_on="assay_id", right_index=True, how="left")
    assay_meta["retained_rows"] = assay_meta["retained_rows"].fillna(0).astype(int)
    return selected.reset_index(drop=True), assay_meta


def main() -> None:
    args = parse_args()
    cache_dir = (REPO_ROOT / args.cache_dir).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_archive = Path(args.raw_activity_candidates).expanduser() if args.raw_activity_candidates else None
    if raw_archive is not None:
        raw_archive = raw_archive if raw_archive.is_absolute() else (REPO_ROOT / raw_archive)
        if not raw_archive.exists():
            raise FileNotFoundError(raw_archive)
        source_fetch_summary = raw_archive.parent / "fetch_summary.csv"
        if not source_fetch_summary.exists():
            raise FileNotFoundError(source_fetch_summary)
        raw = pd.read_csv(raw_archive)
        fetch_summary = pd.read_csv(source_fetch_summary)
    else:
        raw, fetch_summary = collect_activity_rows(args, cache_dir=cache_dir)
    fetch_summary.to_csv(output_dir / "fetch_summary.csv", index=False)
    fetch_coverage = (
        fetch_summary.groupby(["standard_type", "year"], as_index=False)
        .agg(
            fetched_rows=("page_count", "sum"),
            query_total=("total_count", "max"),
            fetch_errors=("error", lambda values: int(pd.Series(values).fillna("").astype(str).str.len().gt(0).sum())),
        )
        .sort_values(["year", "standard_type"])
    )
    fetch_coverage["coverage_fraction"] = fetch_coverage["fetched_rows"] / fetch_coverage["query_total"].replace(0, np.nan)
    fetch_coverage["fetch_complete"] = fetch_coverage["fetched_rows"] >= fetch_coverage["query_total"]
    fetch_coverage.to_csv(output_dir / "fetch_coverage.csv", index=False)
    full_fetch_requested = args.fetch_mode == "full" and args.max_pages_per_year_type == 0
    if full_fetch_requested and not args.allow_incomplete_fetch:
        incomplete = fetch_coverage[(~fetch_coverage["fetch_complete"]) | (fetch_coverage["fetch_errors"] > 0)]
        if not incomplete.empty:
            raise RuntimeError(
                "Complete fetch required, but one or more year/type groups are incomplete. "
                f"See {output_dir / 'fetch_coverage.csv'}"
            )
    if raw_archive is None:
        raw.to_csv(output_dir / "raw_activity_candidates.csv", index=False)
    if raw.empty:
        raise RuntimeError("No activity candidates were collected.")

    collapsed = _collapse_replicates(raw)
    collapsed = _filter_by_group_support(collapsed, "assay_id", args.min_assay_records)

    if args.metadata_mode == "api_verified":
        assay_meta = _fetch_chembl_assay_metadata(
            collapsed["assay_id"],
            cache_dir / "chembl_assay_cache.json",
        )
        collapsed["assay_confidence_score"] = collapsed["assay_id"].map(
            lambda x: int(assay_meta.get(x, {}).get("confidence_score", 0) or 0)
        )
        collapsed["assay_api_text"] = collapsed["assay_id"].map(
            lambda x: assay_meta.get(x, {}).get("text", "")
        )
        collapsed["assay_api_target_id"] = collapsed["assay_id"].map(
            lambda x: assay_meta.get(x, {}).get("target_id", "")
        )
        collapsed["assay_api_type"] = collapsed["assay_id"].map(
            lambda x: assay_meta.get(x, {}).get("assay_type", "")
        )
        collapsed = collapsed[collapsed["assay_confidence_score"] >= args.min_confidence_score].copy()
        collapsed["target_id"] = collapsed["assay_api_target_id"].where(
            collapsed["assay_api_target_id"].fillna("").astype(str).str.len().gt(0),
            collapsed["target_id"],
        )
        collapsed["assay_metadata_source"] = "assay_endpoint"
    else:
        # The full activity response already carries the assay type, free-text description,
        # and target ChEMBL identifier used by the model. Re-querying every assay is neither
        # necessary nor scalable for a complete activity archive.
        collapsed["assay_confidence_score"] = np.nan
        collapsed["assay_api_text"] = collapsed["assay_text"].fillna("")
        collapsed["assay_api_target_id"] = collapsed["target_id"].fillna("")
        collapsed["assay_api_type"] = collapsed["assay_type"].fillna("")
        collapsed["assay_metadata_source"] = "activity_endpoint"

    target_meta = _fetch_chembl_target_metadata(
        collapsed["target_id"],
        cache_dir / "chembl_target_cache.json",
    )
    collapsed["target_accession"] = collapsed["target_id"].map(lambda x: target_meta.get(x, {}).get("accession", ""))
    collapsed["target_type"] = collapsed["target_id"].map(lambda x: target_meta.get(x, {}).get("target_type", ""))
    collapsed["target_pref_name_api"] = collapsed["target_id"].map(lambda x: target_meta.get(x, {}).get("pref_name", ""))
    collapsed = collapsed[
        (collapsed["target_type"] == "SINGLE PROTEIN") & collapsed["target_accession"].fillna("").astype(str).str.len().gt(0)
    ].copy()

    uniprot_meta = _fetch_uniprot_metadata(
        collapsed["target_accession"],
        cache_dir / "chembl_uniprot_cache.json",
    )
    collapsed["sequence"] = collapsed["target_accession"].map(lambda x: uniprot_meta.get(x, {}).get("sequence", ""))
    collapsed["target_text"] = collapsed["target_accession"].map(lambda x: uniprot_meta.get(x, {}).get("text", x))
    collapsed = collapsed[collapsed["sequence"].fillna("").astype(str).str.len().gt(0)].copy()

    if args.label_mode == "absolute_pchembl":
        labeled, threshold_policy = _absolute_pchembl_label(
            collapsed,
            activity_threshold_nm=args.activity_threshold_nm,
        )
        labeled, assay_selection = _assay_balanced_regression_subset(
            labeled,
            max_assays=args.max_assays,
            max_records_per_assay=args.max_records_per_assay,
            seed=args.seed,
        )
        threshold_policy = threshold_policy[
            threshold_policy["assay_id"].isin(set(labeled["assay_id"]))
        ].copy()
    else:
        labeled, threshold_policy = _assay_quantile_label(
            collapsed,
            low_quantile=args.low_quantile,
            high_quantile=args.high_quantile,
            min_records=args.min_assay_records,
            min_gap=args.min_pchembl_gap,
            max_per_class=args.max_per_assay_class,
            seed=args.seed,
        )
        assay_selection = pd.DataFrame()
    if labeled.empty:
        raise RuntimeError("No labeled rows survived assay-aware thresholding.")

    labeled["assay_text"] = labeled["assay_api_text"].where(
        labeled["assay_api_text"].fillna("").astype(str).str.len().gt(0),
        labeled["assay_text"],
    )
    labeled["text"] = (
        labeled["assay_text"].fillna("").astype(str).str.strip()
        + " [SEP] "
        + labeled["target_text"].fillna("").astype(str).str.strip()
    )
    labeled = labeled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    labeled["sample_id"] = [f"v2s{i:07d}" for i in range(len(labeled))]

    columns = [
        "sample_id",
        "drug_id",
        "target_id",
        "assay_id",
        "document_year",
        "standard_type",
        "assay_type",
        "assay_confidence_score",
        "assay_metadata_source",
        "target_accession",
        "target_type",
        "smiles",
        "sequence",
        "assay_text",
        "target_text",
        "text",
        "label",
        "label_raw",
        "pchembl_value",
        "assay_low_pchembl",
        "assay_high_pchembl",
        "assay_threshold_policy",
    ]
    benchmark = labeled[columns].copy()
    benchmark.to_csv(output_dir / "benchmark_frame.csv", index=False)
    threshold_policy.to_csv(output_dir / "threshold_policy_summary.csv", index=False)
    if not assay_selection.empty:
        assay_selection.to_csv(output_dir / "assay_selection_manifest.csv", index=False)

    split_rows: list[dict] = []
    overlap_rows: list[dict] = []
    split_dir = output_dir / f"splits_seed{args.seed}"
    split_dir.mkdir(exist_ok=True)
    for split_mode, builder in _split_builders(args.seed).items():
        frames = builder(benchmark)
        for split_name, split_frame in frames.items():
            split_frame.to_csv(split_dir / f"{split_mode}_{split_name}.csv", index=False)
            split_rows.append(_describe_frame(split_frame, split_mode=split_mode, split_name=split_name))
        overlap_rows.extend(_overlap_rows(split_mode, frames))

    split_summary = pd.DataFrame(split_rows)
    overlap = pd.DataFrame(overlap_rows)
    year_label = (
        benchmark.groupby(["document_year", "label"], as_index=False)
        .size()
        .rename(columns={"size": "rows"})
        .sort_values(["document_year", "label"])
    )
    assay_year = (
        benchmark.groupby("assay_id", as_index=False)
        .agg(
            rows=("sample_id", "size"),
            positive_rate=("label", "mean"),
            min_year=("document_year", "min"),
            max_year=("document_year", "max"),
            unique_drugs=("drug_id", "nunique"),
            unique_targets=("target_id", "nunique"),
            median_pchembl=("pchembl_value", "median"),
        )
        .sort_values("rows", ascending=False)
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)
    overlap.to_csv(output_dir / "split_overlap_audit.csv", index=False)
    year_label.to_csv(output_dir / "label_distribution_by_year.csv", index=False)
    assay_year.to_csv(output_dir / "assay_distribution_summary.csv", index=False)

    overview = {
            "benchmark_name": "CHEMBL_ASSAY",
        "construction": {
            "years": [args.start_year, args.end_year],
            "standard_types": args.standard_types,
            "page_size": args.page_size,
            "fetch_mode": args.fetch_mode,
            "max_pages_per_year_type": args.max_pages_per_year_type,
            "activity_cache_namespace": args.activity_cache_namespace,
            "pages_per_year_type": args.pages_per_year_type,
            "target_organism": args.target_organism,
            "allowed_assay_types": args.allowed_assay_types,
            "min_confidence_score": args.min_confidence_score,
            "metadata_mode": args.metadata_mode,
            "raw_activity_archive": str(raw_archive) if raw_archive is not None else "queried_in_this_run",
            "max_assays": args.max_assays,
            "max_records_per_assay": args.max_records_per_assay,
            "assay_metadata_source": (
                "individual ChEMBL assay endpoint" if args.metadata_mode == "api_verified"
                else "full ChEMBL activity endpoint"
            ),
            "target_type": "SINGLE PROTEIN",
            "label_mode": args.label_mode,
            "binary_label_policy": (
                f"absolute pChEMBL >= {9.0 - math.log10(args.activity_threshold_nm):.4f} "
                f"(<= {args.activity_threshold_nm:g} nM)"
                if args.label_mode == "absolute_pchembl"
                else f"within-assay pChEMBL <= q{args.low_quantile:.2f} inactive, >= q{args.high_quantile:.2f} active"
            ),
            "regression_target": "pchembl_value",
        },
        "fetch_coverage": {
            "groups": int(len(fetch_coverage)),
            "incomplete_groups": int((~fetch_coverage["fetch_complete"]).sum()),
            "fetch_errors": int(fetch_coverage["fetch_errors"].sum()),
            "minimum_coverage_fraction": float(fetch_coverage["coverage_fraction"].min()),
        },
        "raw_activity_candidates": int(len(raw)),
        "post_replicate_collapse_and_assay_support": int(len(_filter_by_group_support(_collapse_replicates(raw), "assay_id", args.min_assay_records))),
        "post_metadata_candidates": int(len(collapsed)),
        "selected_assays": int(labeled["assay_id"].nunique()),
        "dataset_overview": _describe_frame(benchmark),
        "split_summary_rows": int(len(split_summary)),
        "overlap_audit_rows": int(len(overlap)),
    }
    (output_dir / "overview.json").write_text(
        json.dumps(overview, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(overview, indent=2, default=_json_default))


if __name__ == "__main__":
    main()

