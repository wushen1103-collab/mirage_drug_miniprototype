from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "build_chembl_assay_v2.py"
    spec = importlib.util.spec_from_file_location("build_chembl_assay_v2", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(fetch_mode: str):
    return SimpleNamespace(
        start_year=2020,
        end_year=2020,
        standard_types=["IC50"],
        page_size=2,
        fetch_mode=fetch_mode,
        pages_per_year_type=2,
        max_pages_per_year_type=0,
        max_workers=2,
        activity_cache_namespace="test_pages",
        target_organism="Homo sapiens",
        allowed_assay_types=["B", "F"],
        overwrite_fetch_cache=False,
    )


def _payload(offset: int):
    activities = []
    for index in range(offset, min(offset + 2, 5)):
        activities.append(
            {
                "activity_id": index,
                "standard_value": str(10 + index),
                "pchembl_value": str(7.0 - index * 0.1),
                "potential_duplicate": 0,
                "data_validity_comment": None,
                "target_organism": "Homo sapiens",
                "assay_type": "B",
                "canonical_smiles": "CCO",
                "assay_description": "binding assay",
                "molecule_chembl_id": f"CHEMBL{index}",
                "target_chembl_id": "CHEMBLT1",
                "assay_chembl_id": "CHEMBLA1",
                "standard_type": "IC50",
                "document_year": 2020,
                "target_pref_name": "target",
            }
        )
    return {"activities": activities, "page_meta": {"total_count": 5}}


def test_activity_url_pins_order_and_server_side_filters():
    module = _load_script()
    url = module._activity_url(
        "IC50",
        2020,
        1000,
        0,
        target_organism="Homo sapiens",
        allowed_assay_types=["B", "F"],
    )
    query = parse_qs(urlparse(url).query)
    assert query["order_by"] == ["activity_id"]
    assert query["target_organism"] == ["Homo sapiens"]
    assert query["assay_type__in"] == ["B,F"]
    assert query["pchembl_value__isnull"] == ["false"]
    assert query["potential_duplicate"] == ["false"]


def test_full_fetch_mode_discovers_and_fetches_every_page(tmp_path, monkeypatch):
    module = _load_script()
    offsets = []

    def fake_fetch(**kwargs):
        offsets.append(kwargs["offset"])
        return _payload(kwargs["offset"])

    monkeypatch.setattr(module, "_fetch_activity_task", fake_fetch)
    raw, fetch = module.collect_activity_rows(_args("full"), cache_dir=tmp_path)

    assert sorted(offsets) == [0, 2, 4]
    assert len(raw) == 5
    assert fetch["page_count"].sum() == 5
    assert fetch["total_count"].max() == 5


def test_capped_fetch_mode_retains_explicit_pilot_behavior(tmp_path, monkeypatch):
    module = _load_script()
    offsets = []

    def fake_fetch(**kwargs):
        offsets.append(kwargs["offset"])
        return _payload(kwargs["offset"])

    monkeypatch.setattr(module, "_fetch_activity_task", fake_fetch)
    raw, fetch = module.collect_activity_rows(_args("capped"), cache_dir=tmp_path)

    assert sorted(offsets) == [0, 2]
    assert len(raw) == 4
    assert fetch["page_count"].sum() == 4


def test_absolute_pchembl_labels_do_not_depend_on_assay_quantiles():
    module = _load_script()
    frame = module.pd.DataFrame(
        {
            "assay_id": ["A", "A", "B", "B"],
            "pchembl_value": [5.5, 6.2, 5.0, 7.1],
            "drug_id": ["d1", "d2", "d3", "d4"],
            "target_id": ["t1", "t1", "t2", "t2"],
            "document_year": [2020, 2020, 2021, 2021],
        }
    )

    labeled, policy = module._absolute_pchembl_label(frame, activity_threshold_nm=1000.0)

    assert labeled["label"].tolist() == [0, 1, 0, 1]
    assert labeled["assay_threshold_policy"].nunique() == 1
    assert "<= 1000 nM" in labeled["assay_threshold_policy"].iloc[0]
    assert len(policy) == 2


def test_assay_balanced_subset_preserves_continuous_range_without_label_selection():
    module = _load_script()
    rows = []
    for assay_index in range(6):
        for value_index in range(6):
            rows.append(
                {
                    "activity_id": assay_index * 10 + value_index,
                    "assay_id": f"A{assay_index}",
                    "document_year": 2010 + assay_index * 2,
                    "standard_type": "IC50" if assay_index % 2 == 0 else "Ki",
                    "pchembl_value": float(4 + value_index),
                    "label": int(value_index >= 2),
                }
            )
    frame = module.pd.DataFrame(rows)

    selected, manifest = module._assay_balanced_regression_subset(
        frame,
        max_assays=3,
        max_records_per_assay=3,
        seed=42,
    )

    assert manifest["selected"].sum() == 3
    assert selected["assay_id"].nunique() == 3
    assert selected.groupby("assay_id").size().max() == 3
    for _, group in selected.groupby("assay_id"):
        assert group["pchembl_value"].min() == 4.0
        assert group["pchembl_value"].max() == 9.0
