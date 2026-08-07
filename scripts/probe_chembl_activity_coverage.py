from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
ACTIVITY_ENDPOINT = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
STATUS_ENDPOINT = "https://www.ebi.ac.uk/chembl/api/data/status.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe ChEMBL activity counts before a complete, deterministic benchmark download."
    )
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--standard-types", nargs="+", default=["IC50", "Ki", "Kd", "EC50"])
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=12)
    parser.add_argument("--target-organism", default="Homo sapiens")
    parser.add_argument("--allowed-assay-types", nargs="+", default=["B", "F"])
    parser.add_argument("--output-dir", default="outputs/revision_e00_api_coverage_20260729")
    return parser.parse_args()


def activity_url(args: argparse.Namespace, standard_type: str, year: int) -> str:
    params = {
        "limit": 1,
        "offset": 0,
        "standard_units": "nM",
        "standard_relation": "=",
        "standard_type": standard_type,
        "document_year": year,
        "target_organism": args.target_organism,
        "assay_type__in": ",".join(args.allowed_assay_types),
        "pchembl_value__isnull": "false",
        "potential_duplicate": "false",
        "data_validity_comment__isnull": "true",
        "canonical_smiles__isnull": "false",
        "order_by": "activity_id",
    }
    return f"{ACTIVITY_ENDPOINT}?{urlencode(params)}"


def fetch_count(args: argparse.Namespace, standard_type: str, year: int) -> dict:
    url = activity_url(args, standard_type, year)
    response = requests.get(url, timeout=60, headers={"User-Agent": "mirage-revision-audit/1.0"})
    response.raise_for_status()
    payload = response.json()
    total_count = int(payload.get("page_meta", {}).get("total_count") or 0)
    return {
        "standard_type": standard_type,
        "year": year,
        "query_total": total_count,
        "estimated_pages": int(math.ceil(total_count / args.page_size)),
        "query_url": url,
    }


def main() -> None:
    args = parse_args()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (standard_type, year)
        for year in range(args.start_year, args.end_year + 1)
        for standard_type in args.standard_types
    ]
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(fetch_count, args, standard_type, year): (standard_type, year)
            for standard_type, year in tasks
        }
        for future in as_completed(futures):
            standard_type, year = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                rows.append(
                    {
                        "standard_type": standard_type,
                        "year": year,
                        "query_total": 0,
                        "estimated_pages": 0,
                        "query_url": activity_url(args, standard_type, year),
                        "error": repr(exc),
                    }
                )
    coverage = pd.DataFrame(rows).sort_values(["year", "standard_type"]).reset_index(drop=True)
    coverage.to_csv(output_dir / "activity_coverage.csv", index=False)
    status_response = requests.get(STATUS_ENDPOINT, timeout=60, headers={"User-Agent": "mirage-revision-audit/1.0"})
    status_response.raise_for_status()
    summary = {
        "chembl_status": status_response.json(),
        "years": [args.start_year, args.end_year],
        "standard_types": args.standard_types,
        "target_organism": args.target_organism,
        "allowed_assay_types": args.allowed_assay_types,
        "page_size": args.page_size,
        "groups": int(len(coverage)),
        "total_activity_records": int(coverage["query_total"].sum()),
        "estimated_pages": int(coverage["estimated_pages"].sum()),
        "failed_groups": int(coverage.get("error", pd.Series(dtype=str)).notna().sum()),
    }
    (output_dir / "coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
