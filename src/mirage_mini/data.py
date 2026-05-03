from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import requests


UNIPROT_ID_RE = re.compile(r"^[A-Z0-9]{5,10}$")


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    target_text_source: str


def _looks_like_uniprot(accession: str) -> bool:
    return bool(UNIPROT_ID_RE.match(str(accession)))


def _load_json_cache(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json_cache(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_tdc_dataset(dataset_name: str, cache_dir: Path):
    from tdc.multi_pred import DTI

    return DTI(name=dataset_name, path=str(cache_dir))


def _load_tdc_frame(dataset_name: str, cache_dir: Path) -> pd.DataFrame:
    data = _load_tdc_dataset(dataset_name=dataset_name, cache_dir=cache_dir)
    df = data.get_data().copy()
    rename_map = {}
    for old, new in {
        "Drug": "smiles",
        "Target": "sequence",
        "Target_ID": "target_id",
        "Drug_ID": "drug_id",
        "Y": "label_raw",
    }.items():
        if old in df.columns:
            rename_map[old] = new
    df = df.rename(columns=rename_map)
    required = {"smiles", "sequence", "label_raw"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TDC dataset is missing required columns: {sorted(missing)}")
    if "target_id" not in df.columns:
        df["target_id"] = df["sequence"].map(lambda s: f"seq_{abs(hash(s)) % 10**10}")
    if "drug_id" not in df.columns:
        df["drug_id"] = df["smiles"].map(lambda s: f"drug_{abs(hash(s)) % 10**10}")
    return df


def _default_uniprot_text_fetcher(accessions: Iterable[str], cache_path: Path) -> Dict[str, str]:
    requested = [str(x) for x in accessions if pd.notna(x)]
    resolved: Dict[str, str] = {}
    pending: List[str] = []

    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        for accession in requested:
            payload = cache.get(accession)
            if isinstance(payload, str):
                resolved[accession] = payload
            elif isinstance(payload, dict) and payload.get("text"):
                resolved[accession] = str(payload["text"])
            else:
                pending.append(accession)
    else:
        pending = requested

    if pending:
        metadata = _fetch_uniprot_metadata(pending, cache_path=cache_path)
        for key, value in metadata.items():
            if isinstance(value, dict):
                resolved[str(key)] = str(value.get("text", key))
            else:
                resolved[str(key)] = str(value)
    return resolved


def _canonicalize_public_tdc_frame(
    df: pd.DataFrame,
    target_text_map: Dict[str, str],
    activity_threshold_nm: float = 1000.0,
    inactive_threshold_nm: float | None = None,
    greater_is_better: bool = False,
) -> pd.DataFrame:
    rename_map = {}
    for old, new in {
        "Drug": "smiles",
        "Target": "sequence",
        "Target_ID": "target_id",
        "Drug_ID": "drug_id",
        "Y": "label_raw",
    }.items():
        if old in df.columns:
            rename_map[old] = new

    out = df.rename(columns=rename_map).copy()
    required = {"smiles", "sequence", "target_id", "label_raw"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"TDC split is missing required columns: {sorted(missing)}")

    out["target_id"] = out["target_id"].astype(str)
    out["text"] = out["target_id"].map(target_text_map).fillna(out["target_id"])
    return finalize_activity_frame(
        out,
        activity_threshold_nm=activity_threshold_nm,
        inactive_threshold_nm=inactive_threshold_nm,
        greater_is_better=greater_is_better,
    )


def _resolve_tdc_split_request(split_method: str) -> tuple[str, List[str] | None]:
    normalized = split_method.lower().strip()
    if normalized in {"random", "cold_drug", "cold_target"}:
        return normalized, None
    if normalized in {"cold_pair", "cold_drug_target", "pair_cold"}:
        return "cold_split", ["Drug_ID", "Target_ID"]
    raise ValueError(f"Unsupported official TDC split_method: {split_method}")


def _resolve_public_dti_labeling(
    dataset_name: str,
    activity_threshold_nm: float,
    inactive_threshold_nm: float | None,
) -> tuple[float, float | None, bool]:
    normalized = dataset_name.upper().strip()
    if normalized == "KIBA":
        # KIBA is a higher-is-better score, not an nM affinity like DAVIS/BindingDB.
        if activity_threshold_nm == 1000.0 and inactive_threshold_nm is None:
            return 12.8, 11.1, True
        return activity_threshold_nm, inactive_threshold_nm, True
    return activity_threshold_nm, inactive_threshold_nm, False


def prepare_public_dti_official_splits(
    dataset_name: str,
    cache_dir: Path,
    split_method: str,
    seed: int = 42,
    activity_threshold_nm: float = 1000.0,
    inactive_threshold_nm: float | None = None,
    split_frac: tuple[float, float, float] = (0.7, 0.1, 0.2),
    dataset_loader=None,
    text_fetcher=None,
) -> Dict[str, pd.DataFrame]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    loader = dataset_loader or _load_tdc_dataset
    dataset = loader(dataset_name, cache_dir)
    activity_threshold_nm, inactive_threshold_nm, greater_is_better = _resolve_public_dti_labeling(
        dataset_name=dataset_name,
        activity_threshold_nm=activity_threshold_nm,
        inactive_threshold_nm=inactive_threshold_nm,
    )
    tdc_method, column_name = _resolve_tdc_split_request(split_method)
    raw_splits = dataset.get_split(
        method=tdc_method,
        seed=seed,
        frac=list(split_frac),
        column_name=column_name,
    )

    split_frames = {
        "train": raw_splits["train"].copy(),
        "val": raw_splits.get("valid", raw_splits.get("val")).copy(),
        "test": raw_splits["test"].copy(),
    }
    all_target_ids = pd.concat(
        [frame["Target_ID"] for frame in split_frames.values()],
        axis=0,
        ignore_index=True,
    )
    fetch_text = text_fetcher or _default_uniprot_text_fetcher
    target_text_map = fetch_text(
        all_target_ids.astype(str).tolist(),
        cache_path=cache_dir / f"{dataset_name.lower()}_uniprot_text_cache.json",
    )

    return {
        split_name: _canonicalize_public_tdc_frame(
            frame,
            target_text_map=target_text_map,
            activity_threshold_nm=activity_threshold_nm,
            inactive_threshold_nm=inactive_threshold_nm,
            greater_is_better=greater_is_better,
        )
        for split_name, frame in split_frames.items()
    }


def _fetch_uniprot_descriptions(
    target_ids: pd.Series,
    cache_path: Path,
    timeout_sec: int = 20,
) -> Dict[str, str]:
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cache = {}

    session = requests.Session()
    session.headers.update({"User-Agent": "mirage-mini/0.1"})

    descriptions: Dict[str, str] = {}
    unique_ids = sorted({str(x) for x in target_ids if pd.notna(x)})
    for target_id in unique_ids:
        if target_id in cache:
            descriptions[target_id] = cache[target_id]
            continue
        if not _looks_like_uniprot(target_id):
            descriptions[target_id] = target_id
            cache[target_id] = target_id
            continue
        url = f"https://rest.uniprot.org/uniprotkb/{target_id}.json"
        try:
            response = session.get(url, timeout=timeout_sec)
            response.raise_for_status()
            payload = response.json()
            protein_desc = (
                payload.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", "")
            )
            comments = payload.get("comments", [])
            function_bits = []
            for comment in comments:
                if comment.get("commentType") != "FUNCTION":
                    continue
                texts = comment.get("texts", [])
                for text in texts[:2]:
                    value = text.get("value")
                    if value:
                        function_bits.append(value)
                if function_bits:
                    break
            desc = " ".join(x for x in [protein_desc, *function_bits] if x).strip()
            if not desc:
                desc = target_id
        except Exception:
            desc = target_id
        descriptions[target_id] = desc
        cache[target_id] = desc

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return descriptions


def _fetch_uniprot_metadata(
    accessions: Iterable[str],
    cache_path: Path,
    timeout_sec: int = 20,
) -> Dict[str, Dict[str, str]]:
    if cache_path.exists():
        raw_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache = {}
        for key, value in raw_cache.items():
            if isinstance(value, dict):
                cache[str(key)] = {
                    "text": str(value.get("text", key)),
                    "sequence": str(value.get("sequence", "")),
                }
            else:
                cache[str(key)] = {"text": str(value), "sequence": ""}
    else:
        cache = {}

    session = requests.Session()
    session.headers.update({"User-Agent": "mirage-mini/0.1"})

    results: Dict[str, Dict[str, str]] = {}
    pending = sorted({str(x) for x in accessions if pd.notna(x)})
    for accession in pending:
        if accession in cache and not (
            _looks_like_uniprot(accession)
            and isinstance(cache[accession], dict)
            and not cache[accession].get("sequence")
        ):
            results[accession] = cache[accession]
    missing = [accession for accession in pending if accession not in results]

    def _fetch_one(accession: str) -> Tuple[str, Dict[str, str]]:
        if not _looks_like_uniprot(accession):
            return accession, {"text": accession, "sequence": ""}
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
        try:
            response = session.get(url, timeout=timeout_sec)
            response.raise_for_status()
            obj = response.json()
            protein_desc = (
                obj.get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", "")
            )
            sequence = obj.get("sequence", {}).get("value", "")
            comments = obj.get("comments", [])
            function_bits = []
            for comment in comments:
                if comment.get("commentType") != "FUNCTION":
                    continue
                for text in comment.get("texts", [])[:2]:
                    value = text.get("value")
                    if value:
                        function_bits.append(value)
                if function_bits:
                    break
            text = " ".join(x for x in [protein_desc, *function_bits] if x).strip() or accession
            return accession, {"text": text, "sequence": sequence}
        except Exception:
            return accession, {"text": accession, "sequence": ""}

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(_fetch_one, accession) for accession in missing]
        for future in as_completed(futures):
            accession, payload = future.result()
            results[accession] = payload
            cache[accession] = payload

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def finalize_activity_frame(
    df: pd.DataFrame,
    activity_threshold_nm: float = 1000.0,
    inactive_threshold_nm: float | None = None,
    greater_is_better: bool = False,
) -> pd.DataFrame:
    if inactive_threshold_nm is None:
        inactive_threshold_nm = activity_threshold_nm * 10.0

    out = df.dropna(subset=["smiles", "sequence", "text", "target_id", "label_raw"]).copy()
    out["label_raw"] = pd.to_numeric(out["label_raw"], errors="coerce")
    out = out.dropna(subset=["label_raw"]).copy()

    if greater_is_better:
        active = out["label_raw"] >= activity_threshold_nm
        inactive = out["label_raw"] <= inactive_threshold_nm
    else:
        active = out["label_raw"] <= activity_threshold_nm
        inactive = out["label_raw"] >= inactive_threshold_nm
    out = out[active | inactive].copy()
    out["label"] = active.loc[out.index].astype(int)
    out["sample_id"] = [f"s{i:06d}" for i in range(len(out))]
    if "drug_id" not in out.columns:
        out["drug_id"] = out["smiles"].map(lambda s: f"drug_{abs(hash(s)) % 10**10}")
    if "document_year" in out.columns:
        out["document_year"] = pd.to_numeric(out["document_year"], errors="coerce").astype("Int64")
    columns = [
        "sample_id",
        "drug_id",
        "target_id",
        "smiles",
        "sequence",
        "text",
        "label",
        "label_raw",
    ]
    if "assay_id" in out.columns:
        columns.insert(3, "assay_id")
    if "document_year" in out.columns:
        columns.insert(columns.index("smiles"), "document_year")
    if "assay_text" in out.columns:
        columns.insert(columns.index("text"), "assay_text")
    if "target_text" in out.columns:
        columns.insert(columns.index("text") + 1, "target_text")
    return out[columns].reset_index(drop=True)


def prepare_public_dti_subset(
    dataset_name: str,
    cache_dir: Path,
    sample_size: int = 3000,
    max_targets: int = 80,
    activity_threshold_nm: float = 1000.0,
    seed: int = 42,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    df = _load_tdc_frame(dataset_name=dataset_name, cache_dir=cache_dir)
    df = df.dropna(subset=["smiles", "sequence", "target_id", "label_raw"]).copy()
    activity_threshold_nm, inactive_threshold_nm, greater_is_better = _resolve_public_dti_labeling(
        dataset_name=dataset_name,
        activity_threshold_nm=activity_threshold_nm,
        inactive_threshold_nm=None,
    )

    target_counts = df["target_id"].value_counts()
    keep_targets = set(target_counts.head(max_targets).index)
    df = df[df["target_id"].isin(keep_targets)].copy()

    numeric_y = pd.to_numeric(df["label_raw"], errors="coerce")
    df = df[numeric_y.notna()].copy()
    df["label_raw"] = numeric_y.loc[df.index]
    if greater_is_better:
        active = df["label_raw"] >= activity_threshold_nm
        inactive = df["label_raw"] <= inactive_threshold_nm
    else:
        active = df["label_raw"] <= activity_threshold_nm
        inactive = df["label_raw"] >= inactive_threshold_nm
    df = df[active | inactive].copy()
    df["label"] = active.loc[df.index].astype(int)

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed).copy()

    desc_map = _fetch_uniprot_descriptions(
        df["target_id"],
        cache_path=cache_dir / f"{dataset_name.lower()}_uniprot_text_cache.json",
    )
    df["text"] = df["target_id"].map(desc_map).fillna(df["target_id"])
    df["sample_id"] = [f"s{i:06d}" for i in range(len(df))]

    # Add small, realistic missingness into training candidates so the prototype
    # sees incomplete evidence before explicit stress testing.
    mask = rng.random(len(df))
    df.loc[mask < 0.05, "text"] = ""

    columns = [
        "sample_id",
        "drug_id",
        "target_id",
        "smiles",
        "sequence",
        "text",
        "label",
        "label_raw",
    ]
    return DatasetBundle(frame=df[columns].reset_index(drop=True), target_text_source="UniProt")


def _fetch_chembl_json(url: str, timeout_sec: int = 30) -> dict:
    response = requests.get(url, timeout=timeout_sec, headers={"User-Agent": "mirage-mini/0.1"})
    response.raise_for_status()
    return response.json()


def _fetch_chembl_json_cached(
    url: str,
    timeout_sec: int = 30,
    cache_path: Path | None = None,
    max_retries: int = 4,
    retry_backoff_sec: float = 2.0,
) -> dict:
    if cache_path is not None and cache_path.exists():
        cached = _load_json_cache(cache_path, default=None)
        if isinstance(cached, dict):
            return cached

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            payload = _fetch_chembl_json(url, timeout_sec=timeout_sec)
            if cache_path is not None:
                _write_json_cache(cache_path, payload)
            return payload
        except Exception as exc:
            last_exc = exc
            if attempt + 1 < max_retries:
                time.sleep(retry_backoff_sec * (attempt + 1))

    raise RuntimeError(f"Failed to fetch ChEMBL payload after {max_retries} attempts: {url}") from last_exc


def _chembl_activity_page_cache_path(cache_dir: Path, page_size: int, offset: int) -> Path:
    return cache_dir / "chembl_activity_pages" / f"activity_limit{page_size}_offset{offset}.json"


def _chembl_assay_payload_complete(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if int(payload.get("confidence_score") or 0) > 0:
        return True
    return any(str(payload.get(key) or "").strip() for key in ("text", "target_id", "assay_type"))


def _chembl_target_payload_complete(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return any(str(payload.get(key) or "").strip() for key in ("accession", "pref_name", "target_type"))


def _chembl_smiles_complete(payload: object) -> bool:
    return bool(str(payload or "").strip())


def _auto_max_chembl_pages(sample_size: int) -> int:
    if sample_size >= 4000:
        return 72
    if sample_size >= 2000:
        return 48
    if sample_size >= 1000:
        return 24
    return 12


def _collect_chembl_activities(
    sample_size: int,
    cache_dir: Path | None = None,
    max_pages: int = 12,
    page_size: int = 500,
    timeout_sec: int = 30,
) -> pd.DataFrame:
    rows: List[dict] = []
    offset = 0
    page = 0
    while len(rows) < sample_size and page < max_pages:
        url = (
            "https://www.ebi.ac.uk/chembl/api/data/activity.json"
            f"?limit={page_size}&offset={offset}"
        )
        cache_path = None
        if cache_dir is not None:
            cache_path = _chembl_activity_page_cache_path(cache_dir=cache_dir, page_size=page_size, offset=offset)
        try:
            payload = _fetch_chembl_json_cached(
                url,
                timeout_sec=timeout_sec,
                cache_path=cache_path,
            )
        except Exception as exc:
            if not rows:
                raise
            print(f"[chembl] warning: skipping activity page offset={offset}: {exc}")
            offset += page_size
            page += 1
            continue
        activities = payload.get("activities", [])
        if not activities:
            break
        for obj in activities:
            std_type = obj.get("standard_type")
            std_rel = obj.get("standard_relation")
            units = obj.get("standard_units")
            if std_type not in {"IC50", "Ki", "Kd"}:
                continue
            if std_rel != "=":
                continue
            if units != "nM":
                continue
            if not obj.get("molecule_chembl_id") or not obj.get("target_chembl_id") or not obj.get("assay_chembl_id"):
                continue
            value = obj.get("standard_value")
            try:
                value = float(value)
            except Exception:
                continue
            rows.append(
                {
                    "activity_id": obj.get("activity_id"),
                    "assay_id": obj.get("assay_chembl_id"),
                    "target_id": obj.get("target_chembl_id"),
                    "drug_id": obj.get("molecule_chembl_id"),
                    "label_raw": value,
                    "standard_type": std_type,
                    "document_year": obj.get("document_year"),
                }
            )
            if len(rows) >= sample_size:
                break
        offset += page_size
        page += 1
    return pd.DataFrame(rows)


def _fetch_chembl_assay_metadata(
    assay_ids: Iterable[str],
    cache_path: Path,
    timeout_sec: int = 30,
) -> Dict[str, Dict[str, str]]:
    cache = _load_json_cache(cache_path, default={})
    out: Dict[str, Dict[str, str]] = {}
    all_ids = sorted(set(assay_ids))
    for assay_id in all_ids:
        payload = cache.get(assay_id)
        if _chembl_assay_payload_complete(payload):
            out[assay_id] = payload
    missing = [assay_id for assay_id in all_ids if assay_id not in out]

    def _fetch_one(assay_id: str) -> Tuple[str, Dict[str, str] | None]:
        url = f"https://www.ebi.ac.uk/chembl/api/data/assay/{assay_id}.json"
        try:
            obj = _fetch_chembl_json_cached(url, timeout_sec=timeout_sec)
            desc = obj.get("description") or ""
            target_chembl_id = obj.get("target_chembl_id") or ""
            assay_type = obj.get("assay_type") or ""
            confidence = obj.get("confidence_score") or 0
            return assay_id, {
                "text": desc.strip(),
                "target_id": target_chembl_id,
                "assay_type": assay_type,
                "confidence_score": int(confidence),
            }
        except Exception:
            return assay_id, None

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_fetch_one, assay_id) for assay_id in missing]
        for future in as_completed(futures):
            assay_id, payload = future.result()
            if _chembl_assay_payload_complete(payload):
                out[assay_id] = payload
                cache[assay_id] = payload
            else:
                out.setdefault(
                    assay_id,
                    {"text": "", "target_id": "", "assay_type": "", "confidence_score": 0},
                )
    _write_json_cache(cache_path, cache)
    return {assay_id: out[assay_id] for assay_id in all_ids}


def _fetch_chembl_target_metadata(
    target_ids: Iterable[str],
    cache_path: Path,
    timeout_sec: int = 30,
) -> Dict[str, Dict[str, str]]:
    cache = _load_json_cache(cache_path, default={})
    out: Dict[str, Dict[str, str]] = {}
    all_ids = sorted(set(target_ids))
    for target_id in all_ids:
        payload = cache.get(target_id)
        if _chembl_target_payload_complete(payload):
            out[target_id] = payload
    missing = [target_id for target_id in all_ids if target_id not in out]

    def _fetch_one(target_id: str) -> Tuple[str, Dict[str, str] | None]:
        url = f"https://www.ebi.ac.uk/chembl/api/data/target/{target_id}.json"
        try:
            obj = _fetch_chembl_json_cached(url, timeout_sec=timeout_sec)
            components = obj.get("target_components", [])
            accession = ""
            if components:
                accession = components[0].get("accession") or ""
            return target_id, {
                "accession": accession,
                "pref_name": obj.get("pref_name") or "",
                "target_type": obj.get("target_type") or "",
            }
        except Exception:
            return target_id, None

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_fetch_one, target_id) for target_id in missing]
        for future in as_completed(futures):
            target_id, payload = future.result()
            if _chembl_target_payload_complete(payload):
                out[target_id] = payload
                cache[target_id] = payload
            else:
                out.setdefault(target_id, {"accession": "", "pref_name": "", "target_type": ""})
    _write_json_cache(cache_path, cache)
    return {target_id: out[target_id] for target_id in all_ids}


def _fetch_chembl_smiles(
    molecule_ids: Iterable[str],
    cache_path: Path,
    timeout_sec: int = 30,
) -> Dict[str, str]:
    cache = _load_json_cache(cache_path, default={})
    out: Dict[str, str] = {}
    all_ids = sorted(set(molecule_ids))
    for molecule_id in all_ids:
        payload = cache.get(molecule_id)
        if _chembl_smiles_complete(payload):
            out[molecule_id] = str(payload)
    missing = [molecule_id for molecule_id in all_ids if molecule_id not in out]

    def _fetch_one(molecule_id: str) -> Tuple[str, str | None]:
        url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{molecule_id}.json"
        try:
            obj = _fetch_chembl_json_cached(url, timeout_sec=timeout_sec)
            smiles = obj.get("molecule_structures", {}).get("canonical_smiles", "") or ""
        except Exception:
            smiles = None
        return molecule_id, smiles

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(_fetch_one, molecule_id) for molecule_id in missing]
        for future in as_completed(futures):
            molecule_id, smiles = future.result()
            if _chembl_smiles_complete(smiles):
                out[molecule_id] = str(smiles)
                cache[molecule_id] = str(smiles)
            else:
                out.setdefault(molecule_id, "")
    _write_json_cache(cache_path, cache)
    return {molecule_id: out[molecule_id] for molecule_id in all_ids}


def prepare_chembl_assay_subset(
    cache_dir: Path,
    sample_size: int = 1500,
    seed: int = 42,
    activity_threshold_nm: float = 1000.0,
) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = _collect_chembl_activities(
        sample_size=sample_size * 2,
        cache_dir=cache_dir,
        max_pages=_auto_max_chembl_pages(sample_size),
    )
    if raw.empty:
        raise RuntimeError("No ChEMBL activities could be collected")

    assay_meta = _fetch_chembl_assay_metadata(
        raw["assay_id"],
        cache_dir / "chembl_assay_cache.json",
    )
    raw["assay_text"] = raw["assay_id"].map(lambda x: assay_meta.get(x, {}).get("text", ""))
    raw["assay_target_id"] = raw["assay_id"].map(lambda x: assay_meta.get(x, {}).get("target_id", ""))
    raw["assay_type"] = raw["assay_id"].map(lambda x: assay_meta.get(x, {}).get("assay_type", ""))
    raw["confidence_score"] = raw["assay_id"].map(lambda x: assay_meta.get(x, {}).get("confidence_score", 0))

    raw = raw[(raw["confidence_score"] >= 8) & raw["assay_text"].str.len().gt(0)].copy()
    raw["target_id"] = raw["assay_target_id"].where(raw["assay_target_id"].str.len().gt(0), raw["target_id"])

    target_meta = _fetch_chembl_target_metadata(
        raw["target_id"],
        cache_dir / "chembl_target_cache.json",
    )
    raw["target_accession"] = raw["target_id"].map(lambda x: target_meta.get(x, {}).get("accession", ""))
    raw["target_type"] = raw["target_id"].map(lambda x: target_meta.get(x, {}).get("target_type", ""))
    raw = raw[(raw["target_type"] == "SINGLE PROTEIN") & raw["target_accession"].str.len().gt(0)].copy()

    smiles_map = _fetch_chembl_smiles(
        raw["drug_id"],
        cache_dir / "chembl_molecule_cache.json",
    )
    raw["smiles"] = raw["drug_id"].map(smiles_map)
    raw = raw[raw["smiles"].str.len().gt(0)].copy()

    uniprot_meta = _fetch_uniprot_metadata(
        raw["target_accession"],
        cache_dir / "chembl_uniprot_cache.json",
    )
    raw["sequence"] = raw["target_accession"].map(lambda x: uniprot_meta.get(x, {}).get("sequence", ""))
    raw["target_text"] = raw["target_accession"].map(lambda x: uniprot_meta.get(x, {}).get("text", x))
    raw["text"] = (
        raw["assay_text"].fillna("").astype(str).str.strip()
        + " [SEP] "
        + raw["target_text"].fillna("").astype(str).str.strip()
    )
    raw = raw[raw["sequence"].str.len().gt(0)].copy()

    if len(raw) > sample_size:
        raw = raw.sample(n=sample_size, random_state=seed).copy()
    # Small naturalistic missingness
    mask = rng.random(len(raw))
    raw.loc[mask < 0.05, "text"] = ""

    final = finalize_activity_frame(raw, activity_threshold_nm=activity_threshold_nm)
    return DatasetBundle(frame=final, target_text_source="ChEMBL assay text + UniProt")


def prepare_benchmark_splits(
    dataset_name: str,
    cache_dir: Path,
    split_method: str,
    seed: int = 42,
    activity_threshold_nm: float = 1000.0,
    inactive_threshold_nm: float | None = None,
    sample_size: int = 1500,
) -> Dict[str, pd.DataFrame]:
    normalized_dataset = str(dataset_name).upper().strip()
    normalized_split = str(split_method).lower().strip()

    if normalized_dataset == "CHEMBL_ASSAY":
        bundle = prepare_chembl_assay_subset(
            cache_dir=cache_dir,
            sample_size=sample_size,
            seed=seed,
            activity_threshold_nm=activity_threshold_nm,
        )
        if normalized_split == "target_cold":
            return split_target_cold(bundle.frame, seed=seed)
        if normalized_split == "assay_cold":
            return split_assay_cold(bundle.frame, seed=seed)
        if normalized_split == "temporal":
            return split_temporal(bundle.frame, seed=seed)
        if normalized_split == "random":
            return split_random(bundle.frame, seed=seed)
        raise ValueError(f"Unsupported local split_method for CHEMBL_ASSAY: {split_method}")

    return prepare_public_dti_official_splits(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        split_method=split_method,
        seed=seed,
        activity_threshold_nm=activity_threshold_nm,
        inactive_threshold_nm=inactive_threshold_nm,
    )


def split_target_cold(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    return _split_by_group(
        df=df,
        group_column="target_id",
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )


def _split_by_group(
    df: pd.DataFrame,
    group_column: str,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    unique_targets = sorted(df[group_column].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_targets)

    if len(unique_targets) < 3:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(shuffled)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        train = shuffled.iloc[:n_train].reset_index(drop=True)
        val = shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
        test = shuffled.iloc[n_train + n_val :].reset_index(drop=True)
        if len(test) == 0:
            test = val.tail(1).reset_index(drop=True)
            val = val.iloc[:-1].reset_index(drop=True)
        if len(val) == 0:
            val = train.tail(1).reset_index(drop=True)
            train = train.iloc[:-1].reset_index(drop=True)
        return {"train": train, "val": val, "test": test}

    n_targets = len(unique_targets)
    n_train = max(1, int(n_targets * train_frac))
    n_val = max(1, int(n_targets * val_frac))
    train_targets = set(unique_targets[:n_train])
    val_targets = set(unique_targets[n_train : n_train + n_val])
    test_targets = set(unique_targets[n_train + n_val :])
    if not test_targets:
        test_targets = set(sorted(val_targets)[:1])
        val_targets = val_targets - test_targets

    return {
        "train": df[df[group_column].isin(train_targets)].reset_index(drop=True),
        "val": df[df[group_column].isin(val_targets)].reset_index(drop=True),
        "test": df[df[group_column].isin(test_targets)].reset_index(drop=True),
    }


def split_assay_cold(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    if "assay_id" not in df.columns:
        raise ValueError("split_assay_cold requires an assay_id column")
    return _split_by_group(
        df=df,
        group_column="assay_id",
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
    )


def split_temporal(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    year_column: str = "document_year",
) -> Dict[str, pd.DataFrame]:
    if year_column not in df.columns:
        raise ValueError(f"split_temporal requires a {year_column} column")

    working = df.dropna(subset=[year_column]).copy()
    if working.empty:
        raise ValueError(f"split_temporal requires at least one non-null {year_column} value")

    working[year_column] = pd.to_numeric(working[year_column], errors="coerce")
    working = working.dropna(subset=[year_column]).copy()
    if working.empty:
        raise ValueError(f"split_temporal requires numeric {year_column} values")

    working[year_column] = working[year_column].astype(int)
    unique_years = sorted(working[year_column].unique())

    if len(unique_years) < 3:
        ordered = working.sort_values([year_column, "sample_id"], kind="mergesort").reset_index(drop=True)
        n = len(ordered)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        train = ordered.iloc[:n_train].reset_index(drop=True)
        val = ordered.iloc[n_train : n_train + n_val].reset_index(drop=True)
        test = ordered.iloc[n_train + n_val :].reset_index(drop=True)
        if len(test) == 0:
            test = val.tail(1).reset_index(drop=True)
            val = val.iloc[:-1].reset_index(drop=True)
        if len(val) == 0:
            val = train.tail(1).reset_index(drop=True)
            train = train.iloc[:-1].reset_index(drop=True)
        return {"train": train, "val": val, "test": test}

    n_years = len(unique_years)
    n_train = max(1, int(n_years * train_frac))
    n_val = max(1, int(n_years * val_frac))
    if n_train + n_val >= n_years:
        n_val = max(1, n_years - n_train - 1)
    if n_train + n_val >= n_years:
        n_train = max(1, n_years - n_val - 1)

    train_years = set(unique_years[:n_train])
    val_years = set(unique_years[n_train : n_train + n_val])
    test_years = set(unique_years[n_train + n_val :])
    if not test_years:
        moved_year = max(val_years) if val_years else max(train_years)
        test_years = {moved_year}
        val_years = val_years - test_years
        if not val_years:
            train_years = train_years - test_years
            val_years = {max(train_years)}
            train_years = train_years - val_years

    return {
        "train": working[working[year_column].isin(train_years)].reset_index(drop=True),
        "val": working[working[year_column].isin(val_years)].reset_index(drop=True),
        "test": working[working[year_column].isin(test_years)].reset_index(drop=True),
    }


def split_random(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    train = shuffled.iloc[:n_train].reset_index(drop=True)
    val = shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = shuffled.iloc[n_train + n_val :].reset_index(drop=True)
    if len(test) == 0:
        test = val.tail(1).reset_index(drop=True)
        val = val.iloc[:-1].reset_index(drop=True)
    if len(val) == 0:
        val = train.tail(1).reset_index(drop=True)
        train = train.iloc[:-1].reset_index(drop=True)
    return {"train": train, "val": val, "test": test}


def inject_missing_modalities(
    df: pd.DataFrame,
    probs: Dict[str, float],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = df.copy()
    for column, prob in probs.items():
        if column not in out.columns or prob <= 0:
            continue
        mask = rng.random(len(out)) < prob
        out.loc[mask, column] = ""
    return out

