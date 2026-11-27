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

Sinh từng bước (tuần 2, issue #22) — giao diện ở models/base.py:
    state = list, mỗi block một tuple (conv_state, ssm_state); batch ở dim 0, kích thước CỐ ĐỊNH theo số ký tự đã nạp:
        conv_state (B, d_inner, d_conv)  — d_conv đầu vào gần nhất của conv1d (dtype của conv1d.weight)
        ssm_state  (B, d_inner, d_state) — trạng thái ẩn h_t của selective SSM (dtype của dt_proj.weight)
    core_step(h, state): mỗi block h = h + mixer.step(RMSNorm(h)); trả state MỚI, không sửa state truyền vào
        (mamba-ssm cập nhật tại chỗ nên backend cuda clone trước) -> giữ state cũ cho beam search / KSR an toàn.
    prefill(x): nhanh — chạy forward cả đoạn một lần và lấy state cuối (cuda: forward kèm inference_params, kernel
        tự ghi conv_state/ssm_state; torch: scan trả thêm state cuối). Chỉ áp dụng khi nạp từ state rỗng;
        nạp tiếp vào state đã có thì lặp step() như lớp cha. Kết quả bằng bản lặp step() (tests/test_mamba_step.py).
Trích dẫn: state-spaces/mamba (mamba-ssm) — docs/references.md.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseCharLM, RMSNorm, compute_dtype


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

    def forward(self, h: torch.Tensor, return_state: bool = False):
        """h (B, L, d_model) -> out (B, L, d_model); return_state=True trả thêm (conv_state, ssm_state) sau ký tự cuối."""
        L = h.shape[1]
        x, z = self.in_proj(h).chunk(2, dim=-1)                                 # (B, L, di)
        x_in = x.transpose(1, 2)                                                # (B, di, L) — đầu vào conv
        x = F.silu(self.conv1d(x_in)[..., :L].transpose(1, 2))                  # conv nhân quả
        dt, Bm, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                                       # (B, L, di)
        y, last = self._scan(x.float(), dt.float(), -torch.exp(self.A_log.float()), Bm.float(), C.float())
        y = (y + x.float() * self.D.float()).to(z.dtype)
        out = self.out_proj(y * F.silu(z))
        if not return_state:
            return out
        conv_state = F.pad(x_in, (self.d_conv - L, 0))                          # giữ d_conv cột cuối (đệm 0 nếu L < d_conv)
        return out, (conv_state.to(self.conv1d.weight.dtype), last.to(self.dt_proj.weight.dtype))

    def allocate_state(self, batch_size: int, device=None, dtype=None):
        """State rỗng, cùng dạng với mamba_ssm.Mamba.allocate_inference_cache (dtype=None -> dtype trọng số)."""
        device = device if device is not None else self.out_proj.weight.device
        conv = torch.zeros(batch_size, self.d_inner, self.d_conv, device=device, dtype=dtype or self.conv1d.weight.dtype)
        ssm = torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=dtype or self.dt_proj.weight.dtype)
        return conv, ssm

    def step(self, h: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        """1 ký tự: h (B, d_model) -> (out (B, d_model), conv_state mới, ssm_state mới). Không sửa state đầu vào."""
        x, z = self.in_proj(h).chunk(2, dim=-1)                                 # (B, di)
        conv_state = torch.cat([conv_state[..., 1:], x[..., None].to(conv_state.dtype)], dim=-1)   # trượt cửa sổ
        x = F.silu((conv_state * self.conv1d.weight[:, 0, :]).sum(-1) + self.conv1d.bias)
        dt, Bm, C = self.x_proj(x).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).float()                               # (B, di)
        A = -torch.exp(self.A_log.float())                                      # (di, N)
        ssm = ssm_state.float() * torch.exp(dt[..., None] * A) + (dt[..., None] * Bm.float()[:, None, :]) * x.float()[..., None]
        y = torch.einsum("bdn,bn->bd", ssm, C.float()) + x.float() * self.D.float()
        out = self.out_proj(y.to(z.dtype) * F.silu(z))
        return out, conv_state, ssm.to(ssm_state.dtype)

    @staticmethod
    def _scan(x, dt, A, Bm, C):
        """h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t ;  y_t = <h_t, C_t>.  Tuần tự theo L (chậm, chỉ để đối chiếu).
        Trả (y (B, L, di), h_L (B, di, N))."""
        B_, L, di = x.shape
        state = x.new_zeros(B_, di, A.shape[1])
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :, None] * A)                               # (B, di, N)
            state = dA * state + (dt[:, t, :, None] * Bm[:, t, None, :]) * x[:, t, :, None]
            ys.append(torch.einsum("bdn,bn->bd", state, C[:, t]))
        return torch.stack(ys, dim=1), state


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
        self.is_cuda_backend = getattr(cfg, "backend", "cuda") == "cuda"

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        residual = h.float() if self.residual_in_fp32 else h
        return residual + self.mixer(self.norm(h))

    # ---- sinh từng bước ----
    def _state_dtype(self, device) -> torch.dtype:
        # State theo dtype tính toán: fp32 bình thường, fp16 khi đánh giá dưới autocast fp16 — kernel
        # causal_conv1d_update của mamba-ssm bắt buộc conv_state cùng dtype với đầu vào.
        return compute_dtype(device, self.mixer.in_proj.weight.dtype)

    def allocate_state(self, batch_size: int, device=None):
        device = device if device is not None else self.mixer.in_proj.weight.device
        dtype = self._state_dtype(torch.device(device))
        if self.is_cuda_backend:
            return self.mixer.allocate_inference_cache(batch_size, max_seqlen=1, dtype=dtype)   # (conv, ssm)
        return self.mixer.allocate_state(batch_size, device, dtype)

    def step(self, h: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        """h (B, d_model) -> (h mới, conv_state mới, ssm_state mới)."""
        residual = h.float() if self.residual_in_fp32 else h
        x = self.norm(h)
        dtype = self._state_dtype(h.device)
        if self.is_cuda_backend:
            # bản sao (mamba-ssm ghi đè tại chỗ) và đúng dtype tính toán
            conv_state, ssm_state = conv_state.to(dtype, copy=True), ssm_state.to(dtype, copy=True)
            out, conv_state, ssm_state = self.mixer.step(x[:, None], conv_state, ssm_state)
            out = out[:, 0]
        else:
            out, conv_state, ssm_state = self.mixer.step(x, conv_state.to(dtype), ssm_state.to(dtype))
        return residual + out, conv_state, ssm_state

    def forward_with_state(self, h: torch.Tensor):
        """Như forward, trả thêm (conv_state, ssm_state) sau ký tự cuối — dùng cho prefill nhanh."""
        residual = h.float() if self.residual_in_fp32 else h
        x = self.norm(h)
        if self.is_cuda_backend:
            # mamba-ssm: forward kèm inference_params (seqlen_offset = 0) chạy nhánh có return_last_state và ghi
            # conv_state / ssm_state vào key_value_memory_dict[layer_idx]
            params = SimpleNamespace(key_value_memory_dict={}, seqlen_offset=0)
            out = self.mixer(x, inference_params=params)
            state = params.key_value_memory_dict[self.mixer.layer_idx]
        else:
            out, state = self.mixer(x, return_state=True)
        dtype = self._state_dtype(h.device)
        return residual + out, tuple(t.to(dtype) for t in state)


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

    # ---- sinh từng bước (tuần 2, issue #22) ----
    def init_state(self, batch_size: int, device=None):
        device = device if device is not None else self.embedding.weight.device
        return [block.allocate_state(batch_size, device) for block in self.blocks]

    def core_step(self, h: torch.Tensor, state):
        new_state = []
        for block, (conv_state, ssm_state) in zip(self.blocks, state):
            h, conv_state, ssm_state = block.step(h, conv_state, ssm_state)
            new_state.append((conv_state, ssm_state))
        return h, new_state

    @torch.no_grad()
    def prefill(self, x: torch.Tensor, state=None):
        """Nạp x (B, T) từ state rỗng bằng MỘT lần forward (nhanh), trả (logits vị trí cuối, state).
        Nạp tiếp vào state đã có -> lặp step() như BaseCharLM.prefill."""
        if state is not None:
            return super().prefill(x, state)
        h = self.embedding(x)
        new_state = []
        for block in self.blocks:
            h, st = block.forward_with_state(h)
            new_state.append(st)
        return self.lm_head(self.norm_f(h[:, -1])), new_state
