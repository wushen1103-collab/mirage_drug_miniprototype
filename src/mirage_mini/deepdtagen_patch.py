from __future__ import annotations

import torch.nn as nn


def _rebuild_linear(linear: nn.Linear, in_features: int) -> nn.Linear:
    rebuilt = nn.Linear(in_features, linear.out_features, bias=linear.bias is not None)
    nn.init.xavier_uniform_(rebuilt.weight)
    if rebuilt.bias is not None:
        nn.init.zeros_(rebuilt.bias)
    return rebuilt


def patch_decoder_cross_attention_dims(model: nn.Module, *, memory_dim: int) -> int:
    decoder = getattr(model, "decoder", None)
    layers = getattr(decoder, "layer", None)
    if layers is None:
        return 0

    changed = 0
    for layer in layers:
        encoder_attn = getattr(layer, "encoder_attn", None)
        if encoder_attn is None:
            continue
        k_proj = getattr(encoder_attn, "k_proj", None)
        v_proj = getattr(encoder_attn, "v_proj", None)
        if not isinstance(k_proj, nn.Linear) or not isinstance(v_proj, nn.Linear):
            continue
        if k_proj.in_features != memory_dim:
            encoder_attn.k_proj = _rebuild_linear(k_proj, memory_dim)
            changed += 1
        if v_proj.in_features != memory_dim:
            encoder_attn.v_proj = _rebuild_linear(v_proj, memory_dim)
    return changed

