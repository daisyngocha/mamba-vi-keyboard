"""Mamba cài đặt bằng PyTorch thuần (phần "tự cài đặt" để chấm). Owner: Nguyễn Tiến Dũng.

Cấu trúc cần có (theo Section 3 + Figure 3 của paper):
- MambaBlock:
    in_proj: d_model -> 2 * d_inner            (nhánh x và nhánh gate z), d_inner = expand * d_model
    conv1d: depthwise, kernel d_conv, causal (padding = d_conv - 1, cắt phần thừa bên phải)
    x_proj: d_inner -> dt_rank + 2 * d_state   (sinh Δ, B, C phụ thuộc input = selection)
    dt_proj: dt_rank -> d_inner, softplus
    A_log: [d_inner, d_state], khởi tạo S4D-real: A = -[1..d_state]
    D: [d_inner] skip connection
    selective_scan: discretize (ZOH cho A, Euler cho B), quét tuần tự theo t
    out = out_proj(y * silu(z))
- ResidualBlock: x + MambaBlock(RMSNorm(x))
- MambaLM: embedding -> n_layer ResidualBlock -> RMSNorm -> lm_head (tie weight)
- forward(idx[B,T]) -> logits[B,T,V]

Kiểm tra bắt buộc trước khi mở PR:
1. Causality: đổi idx[:, t+1:] không làm đổi logits[:, :t+1].
2. Overfit 1 batch: loss < 0.1 sau ~300 bước.
3. (Nếu cài được mamba_ssm) so khớp output với mamba_ssm.Mamba khi copy trọng số, sai số < 1e-3.
"""
import torch.nn as nn


class MambaLM(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        raise NotImplementedError("TODO(Tiến Dũng): cài đặt MambaLM")

    def forward(self, idx):
        raise NotImplementedError
