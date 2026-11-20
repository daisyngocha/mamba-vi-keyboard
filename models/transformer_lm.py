"""TransformerLM (decoder kiểu GPT). Owner: TF.

Kiến trúc — Mô tả bài toán mục 3, Hướng dẫn Tuần 1 mục 7:
    embedding -> n_layer block pre-norm -> RMSNorm -> lm_head        (phần ngoài lõi ở models/base.py)
    block:  h = h + Attn(RMSNorm(h))    attention nhân quả n_head đầu, RoPE trên q, k
            h = h + MLP(RMSNorm(h))     Linear(d, 4d) -> GELU -> Linear(4d, d)
    Không bias, cùng RMSNorm với MambaLM. Không có learned positional embedding (RoPE ngoại suy được độ dài).
Attention: F.scaled_dot_product_attention(q, k, v, is_causal=True), KHÔNG truyền mask tự tạo. Trên GPU với
cfg.attn_backend == "efficient" ép SDPBackend.EFFICIENT_ATTENTION (bộ nhớ O(L), không rơi về backend math O(L²)).
Khởi tạo kiểu GPT-2: Linear ~ N(0, 0.02), phép chiếu về residual (attn.proj, mlp.fc2) ~ N(0, 0.02 / sqrt(2·n_layer)).

Tuần 2 (issue #23): cài `init_state(batch_size, device)` và `core_step(h, state)` theo giao diện ở models/base.py.
    state = {"pos": số ký tự đã nạp, "kv": [(k, v) mỗi layer, dạng (B, H, pos, head_dim)]}; nối thêm 1 vị trí mỗi bước
    (không cấp phát trước) để state_nbytes phản ánh đúng bộ nhớ KV cache. RoPE dùng offset = pos.
    Tiêu chí đúng: tests/test_step.py (logits từ step() = forward trên cả chuỗi).
Trích dẫn: Su et al. (2021) RoFormer — docs/references.md.
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseCharLM, RMSNorm


def rope_cos_sin(seq_len: int, head_dim: int, base: float, device, offset: int = 0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)                      # (L, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, L, hd). Quy ước rotate-half."""
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


def _sdpa_context(q: torch.Tensor, attn_backend: str):
    if q.is_cuda and attn_backend == "efficient":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    return nullcontext()


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model phải chia hết cho n_head"
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        assert self.head_dim % 2 == 0, "head_dim phải chẵn cho RoPE"
        self.rope_base = getattr(cfg, "rope_base", 10000)
        self.attn_backend = getattr(cfg, "attn_backend", "efficient")
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        q, k, v = self.qkv(h).view(B, L, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)  # 3 × (B, H, L, hd)
        cos, sin = rope_cos_sin(L, self.head_dim, self.rope_base, h.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        with _sdpa_context(q, self.attn_backend):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = getattr(cfg, "mlp_ratio", 4) * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, h):
        return self.fc2(F.gelu(self.fc1(h)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), CausalSelfAttention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.d_model), MLP(cfg)

    def forward(self, h):
        h = h + self.attn(self.norm1(h))
        return h + self.mlp(self.norm2(h))


class TransformerLM(BaseCharLM):
    def build_core(self, cfg) -> None:
        assert getattr(cfg, "pos_encoding", "rope") == "rope", "chỉ hỗ trợ RoPE"
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layer))
        for name, p in self.blocks.named_parameters():
            if p.dim() == 2:
                std = 0.02 / math.sqrt(2 * cfg.n_layer) if name.endswith(("attn.proj.weight", "mlp.fc2.weight")) else 0.02
                nn.init.normal_(p, mean=0.0, std=std)

    def core_forward(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h
