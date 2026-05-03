from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mirage_mini.reporting import (
    compute_selection_instability,
    load_official_run_tables,
    summarize_official_run_tables,
    summarize_selection_instability,
)


STRONGBL_PATTERN = {
    "cold_target": ["tdc_bindingdb_kd_cold_target_fullsuite_strongbaseline_s*_threads64"],
    "cold_drug": ["tdc_bindingdb_kd_cold_drug_fullsuite_strongbaseline_s*_threads64"],
}

ANCHOR_PATTERN = {
    "cold_target": ["tdc_bindingdb_kd_cold_target_anchorfamily_s*_threads64"],
    "cold_drug": ["tdc_bindingdb_kd_cold_drug_anchorfamily_s*_threads64"],
}

ABLATION_ORDER = {
    "cold_target": [
        "retrieval",
        "gated_retrieval",
        "strong_pretrained_gate",
        "gated_strong_pretrained_tuned",
        "interaction_probe_smiles_text_blend",
        "interaction_gate_tuned",
        "hybrid_plus_pretrained_smiles_sequence",
        "sequence_gate_tuned",
        "robust_sequence_anchor",
        "retrieval_sequence_bridge",
        "robust_probe_anchor",
        "fullsuite_val_select",
    ],
    "cold_drug": [
        "retrieval",
        "gated_retrieval",
        "strong_pretrained_gate",
        "gated_strong_pretrained_tuned",
        "interaction_probe_smiles_text_blend",
        "interaction_gate_tuned",
        "hybrid_plus_pretrained_smiles_sequence",
        "sequence_gate_tuned",
        "robust_sequence_anchor",
        "retrieval_sequence_bridge",
        "robust_probe_anchor",
        "fullsuite_val_select",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--output-dir", default="outputs/paper_result_summaries")
    parser.add_argument("--metric-split", default="test_clean", choices=["val", "test_clean", "test_missing"])
    return parser.parse_args()


def _ordered_subset(summary, split_mode: str):
    ordered_models = ABLATION_ORDER[split_mode]
    subset = summary[summary["model"].isin(ordered_models)].copy()
    subset["model"] = subset["model"].astype("category").cat.set_categories(ordered_models, ordered=True)
    return subset.sort_values("model").reset_index(drop=True)


def _top_row(summary):
    ranked = summary.sort_values(["mean_auprc", "mean_auroc"], ascending=[False, False]).reset_index(drop=True)
    return ranked.iloc[0]


def main() -> None:
    args = parse_args()
    outputs_root = REPO_ROOT / args.outputs_root
    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    headline: dict[str, dict] = {}

    for split_mode in ["cold_target", "cold_drug"]:
        strong_runs = load_official_run_tables(
            outputs_root=outputs_root,
            patterns=STRONGBL_PATTERN[split_mode],
            metric_split=args.metric_split,
        )
        strong_summary = summarize_official_run_tables(strong_runs)
        strong_summary.to_csv(output_dir / f"strongbaseline_{split_mode}.csv", index=False)

        anchor_runs = load_official_run_tables(
            outputs_root=outputs_root,
            patterns=ANCHOR_PATTERN[split_mode],
            metric_split=args.metric_split,
        )
        anchor_summary = summarize_official_run_tables(anchor_runs)
        anchor_summary.to_csv(output_dir / f"anchorfamily_{split_mode}.csv", index=False)

        ablation = _ordered_subset(anchor_summary, split_mode=split_mode)
        ablation.to_csv(output_dir / f"ablation_{split_mode}.csv", index=False)

        instability = compute_selection_instability(anchor_runs, selector_model="fullsuite_val_select")
        instability.to_csv(output_dir / f"selection_instability_{split_mode}_per_run.csv", index=False)
        summarize_selection_instability(instability).to_csv(
            output_dir / f"selection_instability_{split_mode}_summary.csv",
            index=False,
        )

        top_strong = _top_row(strong_summary)
        top_anchor = _top_row(anchor_summary)
        headline[split_mode] = {
            "best_strongbaseline_model": top_strong["model"],
            "best_strongbaseline_auprc": float(top_strong["mean_auprc"]),
            "best_strongbaseline_auroc": float(top_strong["mean_auroc"]),
            "best_anchorfamily_model": top_anchor["model"],
            "best_anchorfamily_auprc": float(top_anchor["mean_auprc"]),
            "best_anchorfamily_auroc": float(top_anchor["mean_auroc"]),
            "delta_auprc": float(top_anchor["mean_auprc"] - top_strong["mean_auprc"]),
            "delta_auroc": float(top_anchor["mean_auroc"] - top_strong["mean_auroc"]),
        }

    (output_dir / "headline_metrics.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()

