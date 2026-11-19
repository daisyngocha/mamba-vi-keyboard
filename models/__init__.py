"""Model registry. Giao diện chung xem models/base.py (Hướng dẫn Tuần 1 mục 3.3).

    model = build_model(cfg.model, vocab_size)     # khởi tạo từ config: d_model, n_layer, + tham số riêng
"""
from __future__ import annotations

from models.base import BaseCharLM


def build_model(cfg_model, vocab_size: int) -> BaseCharLM:
    name = cfg_model.name
    if name == "dummy":
        from models.dummy import DummyLM
        return DummyLM(cfg_model, vocab_size)
    if name == "mamba":
        from models.mamba_lm import MambaLM
        return MambaLM(cfg_model, vocab_size)
    if name == "transformer":
        from models.transformer_lm import TransformerLM
        return TransformerLM(cfg_model, vocab_size)
    raise ValueError(f"Unknown model: {name}")
