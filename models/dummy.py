"""Lõi tối giản (MLP từng vị trí) chỉ để smoke test pipeline. Không dùng cho thí nghiệm.
Lõi không nhìn ký tự trước nên trạng thái sinh từng bước chỉ là bộ đếm vị trí — dùng để kiểm tra khung step()/prefill()."""
import torch
import torch.nn as nn

from models.base import BaseCharLM


class DummyLM(BaseCharLM):
    def build_core(self, cfg) -> None:
        self.core = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU())

    def core_forward(self, h):
        return self.core(h)

    def init_state(self, batch_size, device=None):
        return {"pos": 0}

    def core_step(self, h, state):
        return self.core(h), {"pos": state["pos"] + 1}
