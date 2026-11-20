"""MambaLM. Owner: MB.

Kiến trúc — Mô tả bài toán mục 3, Hướng dẫn Tuần 1 mục 6:
    embedding -> n_layer block [ h = h + Mamba(RMSNorm(h)) ] -> RMSNorm -> lm_head   (phần ngoài lõi ở models/base.py)
    Mixer Mamba (selective SSM, Gu & Dao 2023): d_state = 16, d_conv = 4, expand = 2, dt_rank = ceil(d_model / 16).
    residual_in_fp32: dòng residual giữ ở fp32 kể cả khi autocast fp16.

Backend (cfg.backend), cùng tên và cùng kích thước tham số nên state_dict dùng lẫn được:
    "cuda"  -> mamba_ssm.Mamba (kernel CUDA, quyết định Q7) — dùng cho MỌI số đo tốc độ/bộ nhớ.
    "torch" -> TorchMambaMixer bên dưới: scan tuần tự thuần PyTorch, viết lại theo selective_scan_ref của mamba-ssm.
               Chỉ dùng để chạy test trên CPU và đối chiếu kết quả; KHÔNG dùng để đo tốc độ.
Khởi tạo theo quy ước của mamba-ssm (MambaLMHeadModel._init_weights): bias = 0 (trừ dt_proj.bias), out_proj chia
sqrt(n_layer); A_log và D không chịu weight decay (_no_weight_decay).

Tuần 2: step(token, state) sinh từng bước với trạng thái kích thước cố định (conv_state, ssm_state).
Trích dẫn: state-spaces/mamba (mamba-ssm) — docs/references.md.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseCharLM, RMSNorm


class TorchMambaMixer(nn.Module):
    """Mamba mixer thuần PyTorch, cùng tham số với mamba_ssm.Mamba (v1)."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_min: float = 0.001, dt_max: float = 0.1, dt_init_floor: float = 1e-4):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Khởi tạo dt_proj như mamba-ssm: dt ~ log-uniform [dt_min, dt_max], bias = softplus^-1(dt)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank**-0.5, self.dt_rank**-0.5)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        L = h.shape[1]
        x, z = self.in_proj(h).chunk(2, dim=-1)                                 # (B, L, di)
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :L].transpose(1, 2))    # conv nhân quả
        dt, Bm, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                                       # (B, L, di)
        y = self._scan(x.float(), dt.float(), -torch.exp(self.A_log.float()), Bm.float(), C.float())
        y = (y + x.float() * self.D.float()).to(z.dtype)
        return self.out_proj(y * F.silu(z))

    @staticmethod
    def _scan(x, dt, A, Bm, C):
        """h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t ;  y_t = <h_t, C_t>.  Tuần tự theo L (chậm, chỉ để đối chiếu)."""
        B_, L, di = x.shape
        state = x.new_zeros(B_, di, A.shape[1])
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :, None] * A)                               # (B, di, N)
            state = dA * state + (dt[:, t, :, None] * Bm[:, t, None, :]) * x[:, t, :, None]
            ys.append(torch.einsum("bdn,bn->bd", state, C[:, t]))
        return torch.stack(ys, dim=1)


def make_mixer(cfg, layer_idx: int) -> nn.Module:
    backend = getattr(cfg, "backend", "cuda")
    if backend == "cuda":
        from mamba_ssm import Mamba  # ImportError nếu chưa cài wheel từ Dataset mamba-btl-wheels
        return Mamba(d_model=cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand, layer_idx=layer_idx)
    if backend == "torch":
        return TorchMambaMixer(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand)
    raise ValueError(f"backend phải là 'cuda' hoặc 'torch', nhận '{backend}'")


class MambaBlock(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = make_mixer(cfg, layer_idx)
        self.residual_in_fp32 = getattr(cfg, "residual_in_fp32", True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h.float() if self.residual_in_fp32 else h
        return residual + self.mixer(self.norm(h))


class MambaLM(BaseCharLM):
    def build_core(self, cfg) -> None:
        self.blocks = nn.ModuleList(MambaBlock(cfg, i) for i in range(cfg.n_layer))
        for name, p in self.blocks.named_parameters():
            if name.endswith("bias") and p.dim() == 1 and not getattr(p, "_no_reinit", False) and "conv1d" not in name:
                nn.init.zeros_(p)
            if name.endswith("out_proj.weight"):  # rescale_prenorm_residual, 1 residual mỗi block
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(cfg.n_layer)

    def core_forward(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h
