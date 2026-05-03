from __future__ import annotations

import gzip
import importlib
from pathlib import Path
import re
import sys
import tempfile
import types
from typing import Iterable

import requests
import torch


_ALPHAFOLD_URL_RE = re.compile(
    r"^(?P<prefix>https://alphafold\.ebi\.ac\.uk/files/(?P<slug>AF-[A-Z0-9]+-F\d+)-model_v)(?P<version>\d+)\.(?P<ext>cif(?:\.gz)?|pdb)$"
)


def ensure_transformers_onnx_compat(import_module=importlib.import_module, sys_modules=None):
    if sys_modules is None:
        sys_modules = sys.modules

    try:
        return import_module("transformers.onnx")
    except ModuleNotFoundError:
        module = types.ModuleType("transformers.onnx")

        class OnnxConfig:  # pragma: no cover - trivial compatibility shim
            @property
            def inputs(self):
                return {}

        module.OnnxConfig = OnnxConfig
        sys_modules["transformers.onnx"] = module
        return module


def ensure_transformers_pytorch_utils_compat(pytorch_utils_module=None):
    if pytorch_utils_module is None:
        pytorch_utils_module = importlib.import_module("transformers.pytorch_utils")

    if hasattr(pytorch_utils_module, "find_pruneable_heads_and_indices"):
        return pytorch_utils_module

    def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        heads = set(heads) - set(already_pruned_heads)
        mask = torch.ones(n_heads, head_size, dtype=torch.bool)
        for head in heads:
            head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
            mask[head] = False
        index = torch.arange(n_heads * head_size, dtype=torch.long)[mask.view(-1)]
        return heads, index

    pytorch_utils_module.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices
    return pytorch_utils_module


def ensure_transformers_modeling_utils_compat(pretrained_model_cls=None):
    if pretrained_model_cls is None:
        pretrained_model_cls = importlib.import_module("transformers.modeling_utils").PreTrainedModel

    if hasattr(pretrained_model_cls, "get_head_mask"):
        return pretrained_model_cls

    def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked: bool = False):
        if head_mask is None:
            return [None] * num_hidden_layers

        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        else:
            raise ValueError(f"head_mask must have dim 1 or 2, got {head_mask.dim()}")

        head_mask = head_mask.to(dtype=getattr(self, "dtype", torch.float32))
        if is_attention_chunked:
            head_mask = head_mask.unsqueeze(-1)
        return head_mask

    pretrained_model_cls.get_head_mask = get_head_mask
    return pretrained_model_cls


def canonical_alphafold_url_from_entry_id(entry_id: str, *, version: int = 6, extension: str = "cif") -> str | None:
    match = re.fullmatch(r"AF_AF([A-Z0-9]+)F(\d+)", entry_id)
    if match is None:
        return None
    accession, fragment = match.groups()
    return f"https://alphafold.ebi.ac.uk/files/AF-{accession}-F{fragment}-model_v{version}.{extension}"


def build_alphafold_fallback_urls(
    source_url: str | None,
    *,
    entry_id: str | None = None,
    version_priority: Iterable[int] = (6, 5, 4, 3),
) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    def add(url: str | None) -> None:
        if not url or url in seen:
            return
        seen.add(url)
        urls.append(url)

    parsed = _ALPHAFOLD_URL_RE.match(source_url or "")
    if parsed is not None:
        slug = parsed.group("slug")
        source_version = int(parsed.group("version"))
        source_ext = parsed.group("ext")
        add(source_url)
        if source_ext != "pdb":
            add(f"https://alphafold.ebi.ac.uk/files/{slug}-model_v{source_version}.pdb")
        for version in version_priority:
            for extension in ("cif", "pdb"):
                add(f"https://alphafold.ebi.ac.uk/files/{slug}-model_v{version}.{extension}")
    elif entry_id is not None:
        for version in version_priority:
            for extension in ("cif", "pdb"):
                add(canonical_alphafold_url_from_entry_id(entry_id, version=version, extension=extension))

    return urls


def _assign_single_letter_chain_ids(structure) -> None:
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    chain_index = 0
    for model in structure:
        for chain in model:
            if chain_index >= len(allowed):
                raise ValueError("Too many chains for single-character reassignment.")
            chain.id = allowed[chain_index]
            chain_index += 1


def _convert_cif_bytes_to_pdb(cif_bytes: bytes, out_path: str | Path, entry_id: str) -> None:
    import Bio.PDB

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".cif", delete=False) as handle:
        handle.write(cif_bytes)
        tmp_path = Path(handle.name)
    try:
        parser = Bio.PDB.MMCIFParser()
        structure = parser.get_structure(entry_id, str(tmp_path))
        _assign_single_letter_chain_ids(structure)
        pdb_io = Bio.PDB.PDBIO()
        pdb_io.set_structure(structure)
        pdb_io.save(str(out_path))
    finally:
        tmp_path.unlink(missing_ok=True)


def download_alphafold_structure_to_pdb(
    *,
    entry_id: str,
    source_url: str | None,
    out_path: str | Path,
    timeout: int = 60,
) -> str:
    candidate_urls = build_alphafold_fallback_urls(source_url, entry_id=entry_id)
    if not candidate_urls:
        raise RuntimeError(f"No AlphaFold download candidates available for {entry_id}.")

    out_path = Path(out_path)
    errors: list[str] = []
    for url in candidate_urls:
        try:
            response = requests.get(url, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            payload = response.content
            if url.endswith(".pdb"):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(payload)
            elif url.endswith(".cif.gz"):
                _convert_cif_bytes_to_pdb(gzip.decompress(payload), out_path, entry_id)
            else:
                _convert_cif_bytes_to_pdb(payload, out_path, entry_id)
            return url
        except Exception as exc:  # pragma: no cover - exercised in remote smoke runs
            errors.append(f"{url}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to download AlphaFold structure for {entry_id}.\n{joined}")


def install_caster_download_patch(
    caster_process_data,
    *,
    request_get=requests.get,
    download_fn=download_alphafold_structure_to_pdb,
) -> None:
    if getattr(caster_process_data, "_mirage_download_patch_installed", False):
        return

    original = caster_process_data._select_and_download_pdb
    original_get_rcsb_res = getattr(caster_process_data, "get_rcsb_res", None)

    if original_get_rcsb_res is not None:

        def patched_get_rcsb_res(prot_seq, query_type="experimental", allow_complex=False):
            prot_seq_str = "" if prot_seq is None else str(prot_seq).strip()
            if not prot_seq_str or prot_seq_str.lower() == "nan":
                return []
            try:
                return original_get_rcsb_res(prot_seq, query_type=query_type, allow_complex=allow_complex)
            except Exception as exc:  # pragma: no cover - exercised in remote runs
                print(
                    f"[mirage-caster] RCSB {query_type} query failed for sequence length "
                    f"{len(prot_seq_str)}: {exc}",
                    flush=True,
                )
                return []

        caster_process_data.get_rcsb_res = patched_get_rcsb_res

    def patched_select_and_download_pdb(pdb_list, out_path, result_ver="experimental", also_save_accession=True):
        if result_ver != "computational":
            try:
                return original(
                    pdb_list,
                    out_path,
                    result_ver=result_ver,
                    also_save_accession=also_save_accession,
                )
            except Exception as exc:  # pragma: no cover - exercised in remote runs
                print(
                    f"[mirage-caster] Failed experimental download for {pdb_list}: {exc}",
                    flush=True,
                )
                return None

        try:
            if len(pdb_list) == 1:
                pdb_base_identifier = pdb_list[0]
            else:
                pdb_base_identifier = caster_process_data._select_computational_pdb(pdb_list)

            if not pdb_base_identifier:
                print(
                    f"[mirage-caster] No valid AlphaFoldDB candidate remained for {pdb_list}.",
                    flush=True,
                )
                return None

            entry_id, _ = pdb_base_identifier.rsplit("_", 1)
            data_api_url = f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
            data_api_resp = request_get(data_api_url, timeout=60)
            data_api_resp.raise_for_status()
            data_api_resp_json = data_api_resp.json()
            source_url = data_api_resp_json.get("rcsb_comp_model_provenance", {}).get("source_url")

            download_fn(
                entry_id=entry_id,
                source_url=source_url,
                out_path=out_path,
            )

            if also_save_accession:
                accession_fpath = str(out_path).replace(".pdb", "_accession.txt")
                with open(accession_fpath, "w", encoding="utf-8") as handle:
                    handle.write(f"Downloaded from PDB with accession: {pdb_base_identifier}")

            return pdb_base_identifier
        except Exception as exc:  # pragma: no cover - exercised in remote smoke runs
            print(
                f"[mirage-caster] Failed computational download for {pdb_list}: {exc}",
                flush=True,
            )
            return None

    caster_process_data._select_and_download_pdb = patched_select_and_download_pdb
    caster_process_data._mirage_download_patch_installed = True

