"""Phần dùng chung của 2 model (TN): embedding -> lõi -> RMSNorm cuối -> lm_head.

Hai model chỉ khác phần LÕI (Mô tả bài toán mục 3), nên lớp con chỉ cài `build_core` và `core_forward`.
Giao diện — Hướng dẫn Tuần 1 mục 3.3:
    model = build_model(cfg.model, vocab_size)
    logits = model(x)           # x: int64 (B, L) -> logits (B, L, vocab_size); loss tính trong train.py
    model.num_params()          # tổng tham số (embedding và lm_head dùng chung trọng số chỉ đếm 1 lần)

Sinh từng bước (tuần 2, issue #22 MB, #23 TF) — lớp con cài `init_state` và `core_step`:
    state = model.init_state(batch_size, device)      # trạng thái rỗng (Mamba: conv/ssm state; Transformer: KV cache)
    logits, state = model.step(token, state)           # token: int64 (B,) -> logits (B, vocab_size)
    logits, state = model.prefill(x)                   # x: (B, T) -> logits ở vị trí cuối; mặc định lặp step()
    model.state_nbytes(state)                          # số byte của trạng thái (RQ3: bộ nhớ khi gợi ý)
    state = model.reorder_state(state, idx)            # chọn lại / nhân bản chuỗi theo idx (beam search của ĐG)
Quy ước state (để reorder_state dùng chung được): mọi tensor trong state có chiều batch ở dim 0;
giá trị không phải tensor (vd vị trí hiện tại của Transformer) là số dùng chung cho cả batch.
Tiêu chí đúng (tests/test_step.py): logits từ step() ở mọi vị trí t = model(x)[:, t], sai số ≤ 1e-3 (fp32).
step() chạy ở chế độ suy luận: gọi trong torch.no_grad() / inference_mode, model.eval().
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def compute_dtype(device: torch.device, default: torch.dtype) -> torch.dtype:
    """dtype tính toán thực tế: dtype của autocast nếu đang bật (vd đánh giá fp16), ngược lại `default`
    (dtype trọng số). Dùng để cấp phát / ép state sinh từng bước cùng dtype với phép tính."""
    if torch.is_autocast_enabled(device.type):
        return torch.get_autocast_dtype(device.type)
    return default


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

    # ---- lớp con cài đặt: huấn luyện ----
    def build_core(self, cfg) -> None:
        raise NotImplementedError

    def core_forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, d_model) -> (B, L, d_model)."""
        raise NotImplementedError

    # ---- lớp con cài đặt: sinh từng bước ----
    def init_state(self, batch_size: int, device: torch.device | str | None = None) -> Any:
        """Trạng thái rỗng cho batch_size chuỗi. Mamba: kích thước cố định; Transformer: KV cache độ dài 0."""
        raise NotImplementedError(f"{type(self).__name__}.init_state() chưa cài (tuần 2)")

    def core_step(self, h: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        """h: (B, d_model) của 1 ký tự mới -> (đầu ra lõi (B, d_model), state mới)."""
        raise NotImplementedError(f"{type(self).__name__}.core_step() chưa cài (tuần 2)")

    # ---- dùng chung ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm_f(self.core_forward(self.embedding(x))))

    def step(self, token: torch.Tensor, state: Any) -> tuple[torch.Tensor, Any]:
        """token: int64 (B,) -> (logits (B, vocab_size), state mới). Cùng embedding / norm / lm_head với forward."""
        h, state = self.core_step(self.embedding(token), state)
        return self.lm_head(self.norm_f(h)), state

    @torch.no_grad()
    def prefill(self, x: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        """Nạp cả chuỗi x (B, T) vào trạng thái, trả logits ở vị trí cuối. Mặc định lặp step();
        lớp con được viết bản nhanh hơn nhưng phải cho cùng kết quả."""
        if state is None:
            state = self.init_state(x.size(0), x.device)
        logits = None
        for t in range(x.size(1)):
            logits, state = self.step(x[:, t], state)
        return logits, state

    @staticmethod
    def state_nbytes(state: Any) -> int:
        """Tổng số byte của mọi tensor trong state (list / tuple / dict lồng nhau)."""
        if state is None:
            return 0
        if isinstance(state, torch.Tensor):
            return state.numel() * state.element_size()
        if isinstance(state, dict):
            return sum(BaseCharLM.state_nbytes(v) for v in state.values())
        if isinstance(state, (list, tuple)):
            return sum(BaseCharLM.state_nbytes(v) for v in state)
        return 0

    @staticmethod
    def reorder_state(state: Any, idx: torch.Tensor | list[int]) -> Any:
        """Chọn lại / nhân bản các chuỗi trong batch theo idx — dùng cho beam search (evaluate/suggest.py).

        idx: chỉ số chuỗi nguồn cho từng vị trí của batch mới, vd [1, 1, 0] -> batch 3 chuỗi, hai chuỗi đầu
        chép từ chuỗi 1. Mọi tensor trong state được index_select theo dim 0 (tạo bản sao, nên core_step cập nhật
        tại chỗ trên một nhánh không ảnh hưởng nhánh khác); giá trị không phải tensor giữ nguyên.
        """
        if isinstance(state, torch.Tensor):
            return state.index_select(0, torch.as_tensor(idx, dtype=torch.long, device=state.device))
        if isinstance(state, dict):
            return {k: BaseCharLM.reorder_state(v, idx) for k, v in state.items()}
        if isinstance(state, (list, tuple)):
            return type(state)(BaseCharLM.reorder_state(v, idx) for v in state)
        return state

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # parameters() đã bỏ trùng trọng số tie
