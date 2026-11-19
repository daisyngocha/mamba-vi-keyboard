"""Bọc package chính thức mamba-ssm (CUDA kernel) theo interface chung. Owner: Nguyễn Tiến Dũng.

Cài trên Kaggle (GPU): pip install mamba-ssm causal-conv1d
Nếu build lỗi quá 2 giờ -> dừng, ghi lại lỗi vào issue, nhóm dùng mamba_pt.
"""
import torch.nn as nn


class MambaSSMLM(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        raise NotImplementedError("TODO(Tiến Dũng): bọc mamba_ssm.Mamba theo interface chung")

    def forward(self, idx):
        raise NotImplementedError
