from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


DEEPDTA_PROTEIN_DICT = {
    "A": 1,
    "C": 2,
    "B": 3,
    "E": 4,
    "D": 5,
    "G": 6,
    "F": 7,
    "I": 8,
    "H": 9,
    "K": 10,
    "M": 11,
    "L": 12,
    "O": 13,
    "N": 14,
    "Q": 15,
    "P": 16,
    "S": 17,
    "R": 18,
    "U": 19,
    "T": 20,
    "W": 21,
    "V": 22,
    "Y": 23,
    "X": 24,
    "Z": 25,
}
DEEPDTA_SMILES_DICT = {
    "#": 29,
    "%": 30,
    ")": 31,
    "(": 1,
    "+": 32,
    "-": 33,
    "/": 34,
    ".": 2,
    "1": 35,
    "0": 3,
    "3": 36,
    "2": 4,
    "5": 37,
    "4": 5,
    "7": 38,
    "6": 6,
    "9": 39,
    "8": 7,
    "=": 40,
    "A": 41,
    "@": 8,
    "C": 42,
    "B": 9,
    "E": 43,
    "D": 10,
    "G": 44,
    "F": 11,
    "I": 45,
    "H": 12,
    "K": 46,
    "M": 47,
    "L": 13,
    "O": 48,
    "N": 14,
    "P": 15,
    "S": 49,
    "R": 16,
    "U": 50,
    "T": 17,
    "W": 51,
    "V": 18,
    "Y": 52,
    "[": 53,
    "Z": 19,
    "]": 54,
    "\\": 20,
    "a": 55,
    "c": 56,
    "b": 21,
    "e": 57,
    "d": 22,
    "g": 58,
    "f": 23,
    "i": 59,
    "h": 24,
    "m": 60,
    "l": 25,
    "o": 61,
    "n": 26,
    "s": 62,
    "r": 27,
    "u": 63,
    "t": 28,
    "y": 64,
}
DEEPDTA_MAX_SMILES_LEN = 100
DEEPDTA_MAX_PROTEIN_LEN = 1000


def encode_deepdta_smiles(smiles: str, max_len: int = DEEPDTA_MAX_SMILES_LEN) -> np.ndarray:
    encoded = np.zeros(max_len, dtype=np.int64)
    for idx, token in enumerate(str(smiles)[:max_len]):
        encoded[idx] = DEEPDTA_SMILES_DICT.get(token, 0)
    return encoded


def encode_deepdta_protein(sequence: str, max_len: int = DEEPDTA_MAX_PROTEIN_LEN) -> np.ndarray:
    encoded = np.zeros(max_len, dtype=np.int64)
    for idx, token in enumerate(str(sequence)[:max_len]):
        encoded[idx] = DEEPDTA_PROTEIN_DICT.get(token, 0)
    return encoded


@dataclass
class DeepDTABatch:
    smiles: torch.Tensor
    proteins: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> "DeepDTABatch":
        return DeepDTABatch(
            smiles=self.smiles.to(device),
            proteins=self.proteins.to(device),
            labels=self.labels.to(device),
        )


def build_deepdta_batch(
    frame: pd.DataFrame,
    *,
    max_smiles_len: int = DEEPDTA_MAX_SMILES_LEN,
    max_protein_len: int = DEEPDTA_MAX_PROTEIN_LEN,
) -> DeepDTABatch:
    required = {"sample_id", "Drug", "Target", "Y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"DeepDTA frame is missing columns: {sorted(missing)}")

    smiles = np.stack([encode_deepdta_smiles(smile, max_len=max_smiles_len) for smile in frame["Drug"]], axis=0)
    proteins = np.stack([encode_deepdta_protein(seq, max_len=max_protein_len) for seq in frame["Target"]], axis=0)
    labels = frame["Y"].astype(float).to_numpy(dtype=np.float32, copy=True)
    return DeepDTABatch(
        smiles=torch.tensor(smiles, dtype=torch.long),
        proteins=torch.tensor(proteins, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.float32),
    )


class DeepDTATensorDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        max_smiles_len: int = DEEPDTA_MAX_SMILES_LEN,
        max_protein_len: int = DEEPDTA_MAX_PROTEIN_LEN,
    ):
        batch = build_deepdta_batch(
            frame,
            max_smiles_len=max_smiles_len,
            max_protein_len=max_protein_len,
        )
        self.smiles = batch.smiles
        self.proteins = batch.proteins
        self.labels = batch.labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.smiles[index], self.proteins[index], self.labels[index]


class _DeepDTAConvTower(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 32,
        kernel_size: int = 4,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size=kernel_size)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).transpose(1, 2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return self.pool(x).squeeze(-1)


class DeepDTARegressor(nn.Module):
    def __init__(
        self,
        *,
        smiles_vocab_size: int = len(DEEPDTA_SMILES_DICT),
        protein_vocab_size: int = len(DEEPDTA_PROTEIN_DICT),
        embedding_dim: int = 128,
        num_filters: int = 32,
        smiles_kernel_size: int = 4,
        protein_kernel_size: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.smiles_encoder = _DeepDTAConvTower(
            vocab_size=smiles_vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_size=smiles_kernel_size,
        )
        self.protein_encoder = _DeepDTAConvTower(
            vocab_size=protein_vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            kernel_size=protein_kernel_size,
        )
        encoder_dim = num_filters * 3
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim * 2, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, smiles_tokens: torch.Tensor, protein_tokens: torch.Tensor) -> torch.Tensor:
        smiles_repr = self.smiles_encoder(smiles_tokens)
        protein_repr = self.protein_encoder(protein_tokens)
        joint = torch.cat([smiles_repr, protein_repr], dim=-1)
        return self.mlp(joint).view(-1)

