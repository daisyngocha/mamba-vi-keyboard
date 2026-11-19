# Tham khảo chung

Quy định môn học: mọi mã nguồn, thư viện, dữ liệu dùng lại phải ghi vào đây **ngay khi bắt đầu dùng**, trích dẫn trong báo cáo và nhắc đến khi trình bày. Thêm dòng qua PR, ghi rõ người thêm và dùng ở file nào.

| # | Nguồn | Dùng ở đâu | Người thêm |
|---|---|---|---|
| 1 | Gu, A. & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752 | Toàn bộ đề tài; cân tham số 2 block Mamba ≈ 1 layer Transformer | TN |
| 2 | `mamba-ssm`, `causal-conv1d` — github.com/state-spaces/mamba | `models/mamba_lm.py` (backend cuda) | MB |
| 3 | PyTorch (`scaled_dot_product_attention`, AMP) | Toàn bộ code | TN |
| 4 | HuggingFace `datasets`; Wikipedia tiếng Việt `wikimedia/wikipedia` 20231101.vi (CC BY-SA) | `data/prepare.py` | DL |
| 5 | Su, J. et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding* | `models/transformer_lm.py` | TF |
| 6 | `mamba.py` — github.com/alxndrTL/mamba.py (chỉ khi dùng backend torch) | `models/mamba_lm.py` | MB |
| 7 | Gradio hoặc Streamlit | `demo/app.py` | ĐG |
