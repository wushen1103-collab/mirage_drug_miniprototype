from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.reporting import compare_models, filter_shortlist, load_summary_tables, rank_models_across_conditions


DEFAULT_INPUTS = [
    "outputs/official_assay_cold_s2000_m35_s5_p48/summary.csv",
    "outputs/official_target_cold_s2000_m35_s5_p48/summary.csv",
    "outputs/official_temporal_s2000_m35_s5_p48/summary.csv",
    "outputs/focused_missing_s2000_s5_all3splits_m0_m60/summary.csv",
]

DEFAULT_SHORTLIST = [
    "hybrid_plus_pretrained_smiles_text",
    "hybrid_plus_pretrained_smiles",
    "hybrid_blend_pretrained_smiles_text_tuned",
    "pretrained_smiles_dense",
    "pretrained_text_retrieval",
    "hybrid_plus_pretrained_smiles_text_retrieval",
    "hybrid_plus_pretrained_text_retrieval",
    "hybrid_blend_avg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output-dir", default="outputs/unified_main_model_report")
    parser.add_argument("--main-model", default="hybrid_plus_pretrained_smiles_text")
    parser.add_argument("--baseline-model", default="hybrid_blend_avg")
    parser.add_argument("--shortlist", nargs="+", default=DEFAULT_SHORTLIST)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined = load_summary_tables(args.input)
    ranking = rank_models_across_conditions(combined)
    shortlist = filter_shortlist(combined, args.shortlist)
    main_vs_baseline = compare_models(combined, primary_model=args.main_model, baseline_model=args.baseline_model)

    combined.to_csv(output_dir / "combined_conditions.csv", index=False)
    ranking.to_csv(output_dir / "model_ranking.csv", index=False)
    shortlist.to_csv(output_dir / "shortlist_per_condition.csv", index=False)
    main_vs_baseline.to_csv(output_dir / "main_vs_baseline.csv", index=False)
    (output_dir / "report_config.json").write_text(
        json.dumps(
            {
                "input": args.input,
                "main_model": args.main_model,
                "baseline_model": args.baseline_model,
                "shortlist": args.shortlist,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    top_line = ranking.iloc[0]
    print(
        json.dumps(
            {
                "recommended_main_model": top_line["model"],
                "mean_rank_overall": float(top_line["mean_rank_overall"]),
                "mean_auroc": float(top_line["mean_auroc"]),
                "mean_auprc": float(top_line["mean_auprc"]),
                "mean_ece": float(top_line["mean_ece"]),
                "mean_risk80": float(top_line["mean_risk80"]),
            },
            indent=2,
        )
    )
    print(ranking.head(8).to_string(index=False))


if __name__ == "__main__":
    main()


