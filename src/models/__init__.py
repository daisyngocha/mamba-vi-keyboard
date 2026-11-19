"""Model registry.

INTERFACE CHUNG (mọi model phải tuân theo):
    model = build_model(cfg_model, vocab_size)
    logits = model(idx)        # idx: LongTensor [B, T]  ->  logits: FloatTensor [B, T, vocab_size]

- Model là causal: logits[:, t] chỉ phụ thuộc idx[:, :t+1].
- Không tự tính loss bên trong forward (train.py lo phần đó).
- Constructor nhận (cfg_model: SimpleNamespace, vocab_size: int).
"""
from __future__ import annotations

import torch.nn as nn


def build_model(cfg_model, vocab_size: int) -> nn.Module:
    name = cfg_model.name
    if name == "dummy":
        from .dummy import DummyLM
        return DummyLM(cfg_model, vocab_size)
    if name == "transformer":
        from .transformer import TransformerLM
        return TransformerLM(cfg_model, vocab_size)
    if name == "mamba_pt":
        from .mamba_pt import MambaLM
        return MambaLM(cfg_model, vocab_size)
    if name == "mamba_ssm":
        from .mamba_ssm_wrap import MambaSSMLM
        return MambaSSMLM(cfg_model, vocab_size)
    raise ValueError(f"Unknown model: {name}")
