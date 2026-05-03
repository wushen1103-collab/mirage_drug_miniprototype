from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class InteractionProbe(nn.Module):
    def __init__(
        self,
        smiles_dim: int,
        text_dim: int,
        mask_dim: int,
        aux_dim: int = 0,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.aux_dim = int(aux_dim)
        self.smiles_proj = nn.Sequential(
            nn.Linear(smiles_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        joint_dim = proj_dim * 4 + mask_dim + self.aux_dim
        self.mlp = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        smiles_emb: torch.Tensor,
        text_emb: torch.Tensor,
        masks: torch.Tensor,
        aux: torch.Tensor | None = None,
    ) -> torch.Tensor:
        smiles_latent = self.smiles_proj(smiles_emb)
        text_latent = self.text_proj(text_emb)
        pieces = [
            smiles_latent,
            text_latent,
            smiles_latent * text_latent,
            torch.abs(smiles_latent - text_latent),
            masks,
        ]
        if self.aux_dim:
            if aux is None:
                aux = torch.zeros(
                    (smiles_emb.shape[0], self.aux_dim),
                    dtype=smiles_emb.dtype,
                    device=smiles_emb.device,
                )
            pieces.append(aux)
        joint = torch.cat(pieces, dim=1)
        return self.mlp(joint).squeeze(1)


@dataclass
class InteractionProbeBundle:
    model: InteractionProbe
    device: str
    best_epoch: int
    best_val_auprc: float


def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x, dtype=np.float32))


def _coerce_aux(aux: np.ndarray | None, n_rows: int) -> np.ndarray:
    if aux is None:
        return np.zeros((n_rows, 0), dtype=np.float32)
    aux = np.asarray(aux, dtype=np.float32)
    if aux.ndim == 1:
        aux = aux.reshape(-1, 1)
    if aux.shape[0] != n_rows:
        raise ValueError(f"aux row count mismatch: expected {n_rows}, got {aux.shape[0]}")
    return aux


def _make_loader(
    smiles_emb: np.ndarray,
    text_emb: np.ndarray,
    masks: np.ndarray,
    aux: np.ndarray | None,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        _to_tensor(smiles_emb),
        _to_tensor(text_emb),
        _to_tensor(masks),
        _to_tensor(_coerce_aux(aux, len(y))),
        torch.as_tensor(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=False,
    )


def _predict_tensor(
    model: InteractionProbe,
    smiles_emb: np.ndarray,
    text_emb: np.ndarray,
    masks: np.ndarray,
    aux: np.ndarray | None,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    rows = len(smiles_emb)
    loader = DataLoader(
        TensorDataset(
            _to_tensor(smiles_emb),
            _to_tensor(text_emb),
            _to_tensor(masks),
            _to_tensor(_coerce_aux(aux, rows)),
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=False,
    )
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for smiles_batch, text_batch, mask_batch, aux_batch in loader:
            logits = model(
                smiles_batch.to(device),
                text_batch.to(device),
                mask_batch.to(device),
                aux_batch.to(device),
            )
            rows.append(torch.sigmoid(logits).cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0) if rows else np.zeros((0,), dtype=np.float32)


def train_interaction_probe(
    train_smiles: np.ndarray,
    train_text: np.ndarray,
    train_masks: np.ndarray,
    y_train: np.ndarray,
    val_smiles: np.ndarray,
    val_text: np.ndarray,
    val_masks: np.ndarray,
    y_val: np.ndarray,
    train_aux: np.ndarray | None = None,
    val_aux: np.ndarray | None = None,
    device: str | None = None,
    batch_size: int = 64,
    max_epochs: int = 80,
    patience: int = 12,
    hidden_dim: int = 256,
    proj_dim: int = 128,
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    text_dropout_prob: float = 0.0,
    seed: int = 42,
) -> InteractionProbeBundle:
    torch.manual_seed(seed)
    np.random.seed(seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_aux = _coerce_aux(train_aux, len(y_train))
    val_aux = _coerce_aux(val_aux, len(y_val))

    model = InteractionProbe(
        smiles_dim=int(train_smiles.shape[1]),
        text_dim=int(train_text.shape[1]),
        mask_dim=int(train_masks.shape[1]),
        aux_dim=int(train_aux.shape[1]),
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(resolved_device)

    pos = max(1, int(np.asarray(y_train).sum()))
    neg = max(1, int(len(y_train) - pos))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=resolved_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loader = _make_loader(
        smiles_emb=train_smiles,
        text_emb=train_text,
        masks=train_masks,
        aux=train_aux,
        y=y_train,
        batch_size=batch_size,
        shuffle=True,
    )

    best_score = float("-inf")
    best_epoch = -1
    stale_epochs = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    for epoch in range(max_epochs):
        model.train()
        for smiles_batch, text_batch, mask_batch, aux_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if text_dropout_prob > 0.0:
                drop_mask = torch.rand(text_batch.shape[0]) < float(text_dropout_prob)
                if drop_mask.any():
                    text_batch = text_batch.clone()
                    mask_batch = mask_batch.clone()
                    text_batch[drop_mask] = 0.0
                    mask_batch[drop_mask, -1] = 0.0
            logits = model(
                smiles_batch.to(resolved_device),
                text_batch.to(resolved_device),
                mask_batch.to(resolved_device),
                aux_batch.to(resolved_device),
            )
            loss = criterion(logits, y_batch.to(resolved_device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_prob = _predict_tensor(
            model=model,
            smiles_emb=val_smiles,
            text_emb=val_text,
            masks=val_masks,
            aux=val_aux,
            device=resolved_device,
            batch_size=batch_size,
        )
        score = float(average_precision_score(y_val, val_prob))
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.to(resolved_device)
    model.eval()
    return InteractionProbeBundle(
        model=model,
        device=str(resolved_device),
        best_epoch=best_epoch,
        best_val_auprc=float(best_score),
    )


def predict_interaction_probe(
    bundle: InteractionProbeBundle,
    smiles_emb: np.ndarray,
    text_emb: np.ndarray,
    masks: np.ndarray,
    aux: np.ndarray | None = None,
    device: str | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    resolved_device = torch.device(device or bundle.device)
    bundle.model.to(resolved_device)
    return _predict_tensor(
        model=bundle.model,
        smiles_emb=smiles_emb,
        text_emb=text_emb,
        masks=masks,
        aux=aux,
        device=resolved_device,
        batch_size=batch_size,
    )

