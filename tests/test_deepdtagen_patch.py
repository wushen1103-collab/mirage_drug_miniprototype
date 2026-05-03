from __future__ import annotations

import torch.nn as nn

from mirage_mini.deepdtagen_patch import patch_decoder_cross_attention_dims


class _DummyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(512, 376)
        self.v_proj = nn.Linear(512, 376)


class _DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_attn = _DummyAttention()


class _DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([_DummyLayer(), _DummyLayer()])


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _DummyDecoder()


def test_patch_decoder_cross_attention_dims_rebuilds_kv_projections():
    model = _DummyModel()

    assert model.decoder.layer[0].encoder_attn.k_proj.in_features == 512
    assert model.decoder.layer[0].encoder_attn.v_proj.in_features == 512

    changed = patch_decoder_cross_attention_dims(model, memory_dim=376)

    assert changed == 2
    assert model.decoder.layer[0].encoder_attn.k_proj.in_features == 376
    assert model.decoder.layer[0].encoder_attn.v_proj.in_features == 376
    assert model.decoder.layer[1].encoder_attn.k_proj.in_features == 376
    assert model.decoder.layer[1].encoder_attn.v_proj.in_features == 376

