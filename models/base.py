"""Phần dùng chung của 2 model (TN): embedding -> lõi -> RMSNorm cuối -> lm_head.

Hai model chỉ khác phần LÕI (Mô tả bài toán mục 3), nên lớp con chỉ cài `build_core` và `core_forward`.
Giao diện — Hướng dẫn Tuần 1 mục 3.3:
    model = build_model(cfg.model, vocab_size)
    logits = model(x)           # x: int64 (B, L) -> logits (B, L, vocab_size); loss tính trong train.py
    model.num_params()          # tổng tham số (embedding và lm_head dùng chung trọng số chỉ đếm 1 lần)
    model.step(token, state)    # tuần 2: sinh từng bước (Mamba: trạng thái ẩn; Transformer: KV cache)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class BaseCharLM(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.cfg, self.vocab_size = cfg, vocab_size
        self.embedding = nn.Embedding(vocab_size, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)  # init nhỏ, dùng chung cho cả 2 model
        self.build_core(cfg)
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, vocab_size, bias=False)
        if getattr(cfg, "tie_embeddings", True):
            self.lm_head.weight = self.embedding.weight

    # ---- lớp con cài đặt ----
    def build_core(self, cfg) -> None:
        raise NotImplementedError

    def core_forward(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def step(self, token: torch.Tensor, state=None):
        """Tuần 2. token: int64 (B,) -> (logits (B, V), state mới)."""
        raise NotImplementedError("step() làm ở tuần 2")

    # ---- dùng chung ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm_f(self.core_forward(self.embedding(x))))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # parameters() đã bỏ trùng trọng số tie
