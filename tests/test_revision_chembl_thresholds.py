from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = REPO_ROOT / "scripts" / "run_revision_chembl_thresholds.py"
    spec = importlib.util.spec_from_file_location("run_revision_chembl_thresholds", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_absolute_threshold_uses_pchembl_direction_without_filtering_rows():
    module = _load_script()
    frame = pd.DataFrame({"pchembl_value": [5.0, 6.0, 7.0], "sample_id": ["a", "b", "c"]})

    labeled = module.label_from_threshold(frame, 1000.0)

    assert len(labeled) == 3
    assert labeled["label"].tolist() == [0, 1, 1]
    assert module.pchembl_threshold(100.0) == 7.0
