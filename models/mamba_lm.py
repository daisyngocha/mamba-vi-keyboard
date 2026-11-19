"""MambaLM. Owner: MB (Nguyễn Tiến Dũng).

Đặc tả — Mô tả bài toán mục 3, Hướng dẫn Tuần 1 mục 6:
- Kế thừa models.base.BaseCharLM (embedding, RMSNorm cuối, lm_head đã có sẵn); chỉ cài phần lõi:
      n_layer = 8 block, mỗi block: h = h + Mamba(RMSNorm(h))   (residual, bật residual_in_fp32)
- Mixer Mamba: d_state = 16, d_conv = 4, expand = 2 (configs/mamba.yaml).
- Backend qua cờ cfg.backend, phần còn lại của repo không phải sửa:
      "cuda"  -> mamba_ssm.Mamba (kernel CUDA) — mặc định, dùng cho mọi số đo tốc độ/bộ nhớ
      "torch" -> mamba.py thuần PyTorch (alxndrTL/mamba.py) — chỉ dùng nếu đến N4 không cài được mamba-ssm;
                 khi đó báo cáo phải ghi rõ số liệu tốc độ/bộ nhớ không phản ánh đúng Mamba.
- Cân tham số: chọn d_model để num_params() ≈ 8M và lệch TransformerLM ≤ 5% (tests/test_models.py kiểm tra).
- step(token, state) để giữ chỗ cho tuần 2 (sinh từng bước với trạng thái kích thước cố định).

Cài đặt môi trường (N1–N4): ghi `nvidia-smi`, phiên bản torch/CUDA/Python vào install_notes.txt; cài theo thứ tự
  (1) pip install causal-conv1d mamba-ssm --no-build-isolation  (2) wheel khớp phiên bản từ GitHub Releases
  (3) cài lại torch đúng phiên bản wheel hỗ trợ. Lưu *.whl + install_notes.txt lên Kaggle Dataset mamba-btl-wheels.
Smoke test: forward + backward trên GPU; so output kernel CUDA với selective_scan_ref (sai số rất nhỏ là đạt).
Trích dẫn: state-spaces/mamba (mamba-ssm), alxndrTL/mamba.py nếu dùng -> docs/references.md.
"""
from models.base import BaseCharLM


class MambaLM(BaseCharLM):
    def build_core(self, cfg) -> None:
        raise NotImplementedError("TODO(MB): 8 block RMSNorm + Mamba mixer, backend cuda|torch")

    def core_forward(self, h):
        raise NotImplementedError
