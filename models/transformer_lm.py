"""TransformerLM (decoder kiểu GPT). Owner: TF.

Kiến trúc — Mô tả bài toán mục 3, Hướng dẫn Tuần 1 mục 7:
    embedding -> n_layer block pre-norm -> RMSNorm -> lm_head        (phần ngoài lõi ở models/base.py)
    block:  h = h + Attn(RMSNorm(h))    attention nhân quả n_head đầu, RoPE trên q, k
            h = h + MLP(RMSNorm(h))     Linear(d, 4d) -> GELU -> Linear(4d, d)
    Không bias, cùng RMSNorm với MambaLM. Không có learned positional embedding (RoPE ngoại suy được độ dài).
Attention: F.scaled_dot_product_attention(q, k, v, is_causal=True), KHÔNG truyền mask tự tạo. Trên GPU với
cfg.attn_backend == "efficient" ép SDPBackend.EFFICIENT_ATTENTION (bộ nhớ O(L), không rơi về backend math O(L²)).
Khởi tạo kiểu GPT-2: Linear ~ N(0, 0.02), phép chiếu về residual (attn.proj, mlp.fc2) ~ N(0, 0.02 / sqrt(2·n_layer)).

Sinh từng bước (tuần 2, issue #23) — giao diện ở models/base.py:
    state = {"pos": số ký tự đã nạp (int), "kv": [(k, v) mỗi layer, dạng (B, n_head, pos, head_dim)]}; batch ở dim 0.
    KV cache LỚN DẦN: mỗi bước nối thêm 1 vị trí (không cấp phát trước) để state_nbytes phản ánh đúng bộ nhớ
        = 2 · n_layer · pos · d_model · byte (≈ 6 KB/ký tự ở fp16, 12 KB ở fp32) — đối lập với state cố định của Mamba (RQ3).
    RoPE khi sinh: q, k của ký tự mới xoay theo vị trí TUYỆT ĐỐI offset = pos (không phải 0); k lưu trong cache đã xoay.
    Attention khi step: 1 query nhìn toàn bộ cache (gồm chính nó) -> KHÔNG dùng is_causal (với L_q = 1 is_causal sẽ
        che sai, chỉ giữ vị trí đầu tiên). is_causal chỉ dùng trong forward / prefill (q và k cùng độ dài).
    step() trả state mới (torch.cat tạo tensor mới), không sửa state truyền vào.
    prefill(x) nhanh: một lần forward cả đoạn, giữ k, v từng layer làm cache. Chỉ khi nạp từ state rỗng; nạp tiếp vào
        cache đã có thì lặp step() như lớp cha (is_causal của SDPA không đúng khi L_q < L_k).
    Cắt ngữ cảnh khi văn bản dài hơn L lúc huấn luyện là việc của evaluate/ (eval_spec mục 5), model không tự cắt.
Trích dẫn: Su et al. (2021) RoFormer — docs/references.md.
"""
from __future__ import annotations

import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseCharLM, RMSNorm, compute_dtype


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

    def _qkv(self, h: torch.Tensor, offset: int = 0):
        """h (B, L, D) -> q, k, v (B, H, L, hd); RoPE cho q, k theo vị trí offset .. offset + L - 1."""
        B, L, D = h.shape
        q, k, v = self.qkv(h).view(B, L, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)  # 3 × (B, H, L, hd)
        cos, sin = rope_cos_sin(L, self.head_dim, self.rope_base, h.device, offset=offset)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin), v

    def forward(self, h: torch.Tensor, return_kv: bool = False):
        """h (B, L, D) -> (B, L, D); return_kv=True trả thêm (k, v) đã xoay RoPE để làm KV cache (prefill)."""
        B, L, D = h.shape
        q, k, v = self._qkv(h)
        with _sdpa_context(q, self.attn_backend):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = self.proj(y.transpose(1, 2).reshape(B, L, D))
        return (out, (k, v)) if return_kv else out

    def step(self, h: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, pos: int):
        """1 ký tự ở vị trí pos: h (B, D) -> (out (B, D), k_cache mới, v_cache mới) — cache dài thêm 1."""
        B, D = h.shape
        q, k, v = self._qkv(h[:, None], offset=pos)                            # (B, H, 1, hd)
        k = torch.cat([k_cache, k.to(k_cache.dtype)], dim=2)                   # (B, H, pos + 1, hd)
        v = torch.cat([v_cache, v.to(v_cache.dtype)], dim=2)
        with _sdpa_context(q, self.attn_backend):
            y = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype))   # không is_causal: nhìn cả cache
        return self.proj(y.transpose(1, 2).reshape(B, D)), k, v


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

    def step(self, h: torch.Tensor, kv, pos: int):
        """h (B, D) -> (h mới, (k, v) mới) cho 1 ký tự ở vị trí pos."""
        a, k, v = self.attn.step(self.norm1(h), kv[0], kv[1], pos)
        h = h + a
        return h + self.mlp(self.norm2(h)), (k, v)

    def forward_with_kv(self, h: torch.Tensor):
        """Như forward, trả thêm (k, v) của cả đoạn — dùng cho prefill nhanh."""
        a, kv = self.attn(self.norm1(h), return_kv=True)
        h = h + a
        return h + self.mlp(self.norm2(h)), kv


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

    # ---- sinh từng bước (tuần 2, issue #23) ----
    def init_state(self, batch_size: int, device=None):
        """KV cache rỗng (độ dài 0) cho từng layer."""
        device = device if device is not None else self.embedding.weight.device
        attn = self.blocks[0].attn
        dtype = compute_dtype(torch.device(device), attn.qkv.weight.dtype)   # fp16 khi đánh giá dưới autocast fp16

        def empty():
            return torch.zeros(batch_size, attn.n_head, 0, attn.head_dim, device=device, dtype=dtype)

        return {"pos": 0, "kv": [(empty(), empty()) for _ in self.blocks]}

    def core_step(self, h: torch.Tensor, state):
        pos, new_kv = state["pos"], []
        for block, kv in zip(self.blocks, state["kv"]):
            h, kv = block.step(h, kv, pos)
            new_kv.append(kv)
        return h, {"pos": pos + 1, "kv": new_kv}

    @torch.no_grad()
    def prefill(self, x: torch.Tensor, state=None):
        """Nạp x (B, T) từ cache rỗng bằng MỘT lần forward (nhanh), trả (logits vị trí cuối, state).
        Nạp tiếp vào cache đã có -> lặp step() như BaseCharLM.prefill."""
        if state is not None and state["pos"] > 0:
            return super().prefill(x, state)
        h = self.embedding(x)
        kv = []
        for block in self.blocks:
            h, layer_kv = block.forward_with_kv(h)
            kv.append(layer_kv)
        return self.lm_head(self.norm_f(h[:, -1])), {"pos": x.size(1), "kv": kv}
