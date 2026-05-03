from __future__ import annotations

import importlib

from mirage_mini.external_compat import (
    build_alphafold_fallback_urls,
    canonical_alphafold_url_from_entry_id,
    ensure_transformers_modeling_utils_compat,
    ensure_transformers_onnx_compat,
    ensure_transformers_pytorch_utils_compat,
)


def test_canonical_alphafold_url_from_entry_id_uses_current_version():
    assert canonical_alphafold_url_from_entry_id("AF_AFP51449F1") == "https://alphafold.ebi.ac.uk/files/AF-P51449-F1-model_v6.cif"


def test_build_alphafold_fallback_urls_includes_current_and_legacy_versions():
    urls = build_alphafold_fallback_urls("https://alphafold.ebi.ac.uk/files/AF-P51449-F1-model_v4.cif")

    assert "https://alphafold.ebi.ac.uk/files/AF-P51449-F1-model_v4.cif" in urls
    assert "https://alphafold.ebi.ac.uk/files/AF-P51449-F1-model_v6.cif" in urls
    assert "https://alphafold.ebi.ac.uk/files/AF-P51449-F1-model_v6.pdb" in urls
    assert len(urls) == len(set(urls))


def test_ensure_transformers_onnx_compat_injects_minimal_shim():
    local_modules = {}

    def fake_import(name: str):
        if name == "transformers.onnx":
            raise ModuleNotFoundError(name)
        return importlib.import_module(name)

    module = ensure_transformers_onnx_compat(import_module=fake_import, sys_modules=local_modules)

    assert local_modules["transformers.onnx"] is module
    assert hasattr(module, "OnnxConfig")


def test_ensure_transformers_pytorch_utils_compat_injects_pruning_helper():
    module = type("DummyModule", (), {})()

    ensure_transformers_pytorch_utils_compat(module)
    heads, index = module.find_pruneable_heads_and_indices([1], 4, 2, {3})

    assert heads == {1}
    assert index.tolist() == [0, 1, 4, 5, 6, 7]


def test_ensure_transformers_modeling_utils_compat_injects_get_head_mask():
    cls = type("DummyPreTrainedModel", (), {"dtype": None})

    ensure_transformers_modeling_utils_compat(cls)
    instance = cls()
    head_mask = cls.get_head_mask(instance, None, 3)

    assert head_mask == [None, None, None]

