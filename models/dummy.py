"""Lõi tối giản (MLP từng vị trí) chỉ để smoke test pipeline. Không dùng cho thí nghiệm."""
import torch.nn as nn

from models.base import BaseCharLM


class DummyLM(BaseCharLM):
    def build_core(self, cfg) -> None:
        self.core = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU())

    def core_forward(self, h):
        return self.core(h)
