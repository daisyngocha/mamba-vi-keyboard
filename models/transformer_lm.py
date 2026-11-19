"""TransformerLM (decoder kiểu GPT). Owner: TF (Đặng Duyên Hà).

Đặc tả — Mô tả bài toán mục 3, Hướng dẫn Tuần 1 mục 7:
- Kế thừa models.base.BaseCharLM (embedding, RMSNorm cuối, lm_head đã có sẵn); chỉ cài phần lõi:
      n_layer = 4 block pre-norm, mỗi block:
          h = h + Attn(RMSNorm(h))      # attention nhân quả, n_head đầu, RoPE trên q, k
          h = h + MLP(RMSNorm(h))       # MLP mlp_ratio = 4×
  Dùng models.base.RMSNorm để cùng kiểu norm với MambaLM.
- Mã hóa vị trí RoPE (KHÔNG dùng learned positional embedding) — cần cho thí nghiệm ngoại suy:
  train L = 1.024, đánh giá khi văn bản đã gõ dài 2.048 / 4.096.
- Attention: F.scaled_dot_product_attention(q, k, v, is_causal=True); KHÔNG truyền mask tự tạo
  (mask có thể đẩy PyTorch sang backend math O(L²) bộ nhớ). Với cfg.attn_backend == "efficient":
      from torch.nn.attention import sdpa_kernel, SDPBackend
      with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]): ...
- Cân tham số: chọn d_model để num_params() ≈ 8M và lệch MambaLM ≤ 5%.
- Kiểm tra bắt buộc: tính nhân quả (tests/test_models.py); L = 4.096 với batch 1, 2, 4 trên T4 -> bảng bộ nhớ
  đỉnh (OOM thì ghi nhận, không cố sửa); 500 bước fp16 bằng train.py không NaN, loss giảm.
- Tuần 2: step(token, state) với KV cache; tiêu chí đúng: logits khi sinh có cache = logits khi tính lại cả chuỗi.
"""
from models.base import BaseCharLM


class TransformerLM(BaseCharLM):
    def build_core(self, cfg) -> None:
        raise NotImplementedError("TODO(TF): 4 block pre-norm, attention SDPA nhân quả + RoPE, MLP 4×")

    def core_forward(self, h):
        raise NotImplementedError
