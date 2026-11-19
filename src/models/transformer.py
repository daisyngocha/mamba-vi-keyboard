"""GPT-style char-level Transformer. Owner: Đặng Duyên Hà.

Yêu cầu:
- Decoder-only, pre-LayerNorm, causal self-attention (dùng F.scaled_dot_product_attention(is_causal=True)).
- Learned positional embedding, độ dài tối đa = cfg.max_seq_len (cần 4096 cho benchmark E3).
- Tham số cấu hình: d_model, n_layer, n_head, ffn_mult, dropout.
- Weight tying giữa token embedding và lm_head.
- forward(idx[B,T]) -> logits[B,T,V]  (xem interface trong src/models/__init__.py)
"""
import torch.nn as nn


class TransformerLM(nn.Module):
    def __init__(self, cfg, vocab_size: int):
        super().__init__()
        raise NotImplementedError("TODO(Duyên Hà): cài đặt TransformerLM")

    def forward(self, idx):
        raise NotImplementedError
