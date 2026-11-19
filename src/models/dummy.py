"""Model tối giản (bigram + MLP) chỉ để smoke test pipeline. Không dùng cho thí nghiệm."""
import torch.nn as nn


class DummyLM(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.GELU())
        self.head = nn.Linear(cfg.d_model, vocab_size)

    def forward(self, idx):
        return self.head(self.mlp(self.emb(idx)))
