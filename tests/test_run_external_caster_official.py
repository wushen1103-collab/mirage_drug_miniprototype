from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_external_caster_official.py"
SPEC = importlib.util.spec_from_file_location("run_external_caster_official", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_resolve_caster_pool_workers_caps_parallelism_to_safe_value():
    workers = MODULE.resolve_caster_pool_workers(
        dataloader_workers=8,
        cpu_count=384,
        fd_soft_limit=1024,
    )

    assert workers == 8


def test_resolve_caster_pool_workers_never_returns_zero():
    workers = MODULE.resolve_caster_pool_workers(
        dataloader_workers=0,
        cpu_count=384,
        fd_soft_limit=1024,
    )

    assert workers == 1

