from __future__ import annotations

import pandas as pd

from mirage_mini.reporting import compare_models, filter_shortlist, load_summary_tables, rank_models_across_conditions


def test_load_summary_tables_combines_and_deduplicates(tmp_path):
    frame = pd.DataFrame(
        {
            "split_mode": ["assay_cold", "assay_cold"],
            "missing_sequence_prob": [0.35, 0.35],
            "missing_text_prob": [0.35, 0.35],
            "model": ["a", "b"],
            "mean_test_missing_auroc": [0.8, 0.7],
            "mean_test_missing_auprc": [0.9, 0.8],
            "mean_test_missing_ece": [0.1, 0.2],
            "mean_test_missing_risk80": [0.12, 0.15],
        }
    )
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    frame.to_csv(first, index=False)
    frame.iloc[[0]].to_csv(second, index=False)

    combined = load_summary_tables([first, second])

    assert len(combined) == 2
    assert set(combined["model"]) == {"a", "b"}


def test_rank_models_across_conditions_prefers_stronger_and_better_calibrated_models():
    summary = pd.DataFrame(
        {
            "split_mode": ["assay_cold", "assay_cold", "temporal", "temporal"],
            "missing_sequence_prob": [0.35, 0.35, 0.35, 0.35],
            "missing_text_prob": [0.35, 0.35, 0.35, 0.35],
            "model": ["main", "base", "main", "base"],
            "mean_test_missing_auroc": [0.82, 0.75, 0.61, 0.58],
            "mean_test_missing_auprc": [0.91, 0.88, 0.89, 0.87],
            "mean_test_missing_ece": [0.07, 0.11, 0.08, 0.12],
            "mean_test_missing_risk80": [0.10, 0.14, 0.11, 0.15],
        }
    )

    ranked = rank_models_across_conditions(summary)

    assert ranked.iloc[0]["model"] == "main"
    assert ranked.iloc[0]["mean_rank_overall"] < ranked.iloc[1]["mean_rank_overall"]


def test_filter_shortlist_sorts_within_conditions():
    summary = pd.DataFrame(
        {
            "split_mode": ["temporal", "temporal", "assay_cold"],
            "missing_sequence_prob": [0.6, 0.6, 0.0],
            "missing_text_prob": [0.6, 0.6, 0.0],
            "model": ["m2", "m1", "m1"],
            "mean_test_missing_auroc": [0.56, 0.58, 0.85],
            "mean_test_missing_auprc": [0.88, 0.89, 0.96],
            "mean_test_missing_ece": [0.09, 0.08, 0.07],
            "mean_test_missing_risk80": [0.12, 0.11, 0.09],
        }
    )

    selected = filter_shortlist(summary, ["m1", "m2"])

    assert list(selected["model"]) == ["m1", "m1", "m2"]


def test_compare_models_computes_conditionwise_deltas():
    summary = pd.DataFrame(
        {
            "split_mode": ["assay_cold", "assay_cold", "temporal", "temporal"],
            "missing_sequence_prob": [0.35, 0.35, 0.6, 0.6],
            "missing_text_prob": [0.35, 0.35, 0.6, 0.6],
            "model": ["main", "base", "main", "base"],
            "mean_test_missing_auroc": [0.82, 0.80, 0.58, 0.55],
            "mean_test_missing_auprc": [0.91, 0.90, 0.88, 0.87],
            "mean_test_missing_ece": [0.07, 0.08, 0.06, 0.09],
            "mean_test_missing_risk80": [0.10, 0.12, 0.11, 0.13],
        }
    )

    comparison = compare_models(summary, primary_model="main", baseline_model="base")

    assert len(comparison) == 2
    assert (comparison["delta_auroc"] > 0).all()
    assert (comparison["delta_auprc"] > 0).all()
    assert (comparison["delta_ece"] < 0).all()
    assert (comparison["delta_risk80"] < 0).all()


