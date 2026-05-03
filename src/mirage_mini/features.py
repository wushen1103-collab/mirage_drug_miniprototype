from __future__ import annotations

from dataclasses import dataclass
import pickle
import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.feature_extraction.text import HashingVectorizer


@dataclass
class ModalFeatures:
    matrix: sparse.csr_matrix
    masks: np.ndarray
    feature_names: List[str]
    smiles_matrix: sparse.csr_matrix
    sequence_matrix: sparse.csr_matrix
    text_matrix: sparse.csr_matrix


class MultiModalFeaturizer:
    def __init__(
        self,
        smiles_features: int = 512,
        sequence_features: int = 512,
        text_features: int = 256,
        include_masks: bool = True,
    ) -> None:
        self.include_masks = include_masks
        self.smiles_vectorizer = HashingVectorizer(
            n_features=smiles_features,
            analyzer="char",
            ngram_range=(2, 4),
            alternate_sign=False,
            norm="l2",
            lowercase=False,
        )
        self.sequence_vectorizer = HashingVectorizer(
            n_features=sequence_features,
            analyzer="char",
            ngram_range=(3, 5),
            alternate_sign=False,
            norm="l2",
            lowercase=False,
        )
        self.text_vectorizer = HashingVectorizer(
            n_features=text_features,
            analyzer="word",
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
            lowercase=True,
        )
        self.feature_names = [
            *(f"smiles_{i}" for i in range(smiles_features)),
            *(f"sequence_{i}" for i in range(sequence_features)),
            *(f"text_{i}" for i in range(text_features)),
        ]
        if self.include_masks:
            self.feature_names.extend(["mask_smiles", "mask_sequence", "mask_text"])

    @staticmethod
    def _presence_bits(series: pd.Series) -> np.ndarray:
        return series.fillna("").astype(str).str.len().gt(0).astype(np.float32).to_numpy()

    def transform(self, df: pd.DataFrame) -> ModalFeatures:
        smiles = df["smiles"].fillna("").astype(str).tolist()
        sequence = df["sequence"].fillna("").astype(str).tolist()
        text = df["text"].fillna("").astype(str).tolist()

        smiles_x = self.smiles_vectorizer.transform(smiles)
        sequence_x = self.sequence_vectorizer.transform(sequence)
        text_x = self.text_vectorizer.transform(text)
        masks = np.stack(
            [
                self._presence_bits(df["smiles"]),
                self._presence_bits(df["sequence"]),
                self._presence_bits(df["text"]),
            ],
            axis=1,
        )
        if self.include_masks:
            mask_x = sparse.csr_matrix(masks)
            matrix = sparse.hstack([smiles_x, sequence_x, text_x, mask_x], format="csr")
        else:
            matrix = sparse.hstack([smiles_x, sequence_x, text_x], format="csr")
        return ModalFeatures(
            matrix=matrix,
            masks=masks,
            feature_names=self.feature_names,
            smiles_matrix=smiles_x.tocsr(),
            sequence_matrix=sequence_x.tocsr(),
            text_matrix=text_x.tocsr(),
        )


class MorganFingerprintFeaturizer:
    def __init__(self, n_bits: int = 2048, radius: int = 2) -> None:
        self.n_bits = n_bits
        self.radius = radius

    def _smiles_to_fp(self, smiles: str) -> np.ndarray:
        array = np.zeros((self.n_bits,), dtype=np.float32)
        mol = Chem.MolFromSmiles(str(smiles)) if smiles else None
        if mol is None:
            return array
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        DataStructs.ConvertToNumpyArray(fp, array)
        return array

    def transform(self, smiles: Iterable[str]) -> sparse.csr_matrix:
        rows = [self._smiles_to_fp(x) for x in smiles]
        if not rows:
            return sparse.csr_matrix((0, self.n_bits), dtype=np.float32)
        dense = np.vstack(rows).astype(np.float32)
        return sparse.csr_matrix(dense)


def _slugify_model_name(model_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()
    return slug or "model"


class _HuggingFaceEmbeddingBackend:
    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        max_length: int = 128,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.max_length = max_length
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[np.ndarray] = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                tokenized = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokenized = {key: value.to(self.device) for key, value in tokenized.items()}
                outputs = self.model(**tokenized)
                hidden = outputs.last_hidden_state
                mask = tokenized["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                rows.append(pooled.float().cpu().numpy().astype(np.float32))
        return np.vstack(rows)


class CachedTransformerEmbedder:
    def __init__(
        self,
        model_name: str,
        cache_path: Path | None = None,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = 128,
        normalize: bool = True,
        backend=None,
    ) -> None:
        self.model_name = model_name
        self.cache_path = cache_path or Path(".cache") / f"{_slugify_model_name(model_name)}_embeddings.pkl"
        self.batch_size = batch_size
        self.device = device
        self.max_length = max_length
        self.normalize = normalize
        self._backend = backend
        self._cache_loaded = False
        self._cache: dict[str, np.ndarray] = {}
        self.dim = int(getattr(backend, "dim", 0)) if backend is not None else 0

    def _ensure_backend(self):
        if self._backend is None:
            self._backend = _HuggingFaceEmbeddingBackend(
                model_name=self.model_name,
                device=self.device,
                max_length=self.max_length,
            )
            self.dim = int(self._backend.dim)
        return self._backend

    def _load_cache(self) -> None:
        if self._cache_loaded:
            return
        if self.cache_path.exists():
            with self.cache_path.open("rb") as handle:
                raw_cache = pickle.load(handle)
            self._cache = {str(key): np.asarray(value, dtype=np.float32) for key, value in raw_cache.items()}
            if self._cache and self.dim == 0:
                self.dim = int(next(iter(self._cache.values())).shape[0])
        self._cache_loaded = True

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as handle:
            pickle.dump(self._cache, handle)

    def _normalize_rows(self, matrix: np.ndarray) -> np.ndarray:
        if not self.normalize or matrix.size == 0:
            return matrix.astype(np.float32, copy=False)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            np.clip(norms, a_min=1e-8, a_max=None),
            out=np.zeros_like(matrix, dtype=np.float32),
            where=norms > 0,
        ).astype(np.float32, copy=False)

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        self._load_cache()
        values = [str(x).strip() if x is not None else "" for x in texts]
        missing = sorted({value for value in values if value and value not in self._cache})

        if missing:
            backend = self._ensure_backend()
            encoded = backend.encode(missing, batch_size=self.batch_size)
            encoded = self._normalize_rows(np.asarray(encoded, dtype=np.float32))
            self.dim = int(encoded.shape[1]) if encoded.ndim == 2 else self.dim
            for key, row in zip(missing, encoded):
                self._cache[key] = row.astype(np.float32, copy=False)
            self._save_cache()
        elif self.dim == 0:
            self._ensure_backend()

        if self.dim == 0:
            self.dim = 1

        output = np.zeros((len(values), self.dim), dtype=np.float32)
        for idx, value in enumerate(values):
            if value and value in self._cache:
                output[idx] = self._cache[value]
        return output
