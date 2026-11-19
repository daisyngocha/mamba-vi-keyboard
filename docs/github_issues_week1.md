# GitHub Issues — Tuần 1 (20–26/11, N1–N7)

Nội dung các issue tuần 1, viết lại theo *Hướng dẫn công việc Tuần 1 theo vai trò (v1)*. Cập nhật lại nội dung issue #2–#6 trên GitHub theo file này.
Label: `data` / `model` / `train-eval` / `lead` · Milestone: `Tuần 1 – nền tảng dùng chung`.
Nhánh: `<mã>/<việc>` (vd `dl/prepare`, `mb/mamba-lm`). Mốc in đậm là điểm người khác đang chờ.

| Mã | Vai trò | Thành viên |
|---|---|---|
| TN | Trưởng nhóm | Lê Thị Ngọc Hà |
| DL | Dữ liệu | Ngô Thị Thu Hằng |
| MB | Mô hình Mamba | Nguyễn Tiến Dũng |
| TF | Mô hình Transformer | Đặng Duyên Hà |
| ĐG | Huấn luyện & đánh giá | Cao Thanh Hằng |

---

### #2 [DL] Dữ liệu Wikipedia tiếng Việt + loader
**Nhánh:** `dl/prepare`, `dl/loader` · **Mốc:** **sample.bin N2 (21/11)**, **bộ đầy đủ N5 (24/11)**
Đặc tả trong docstring `data/prepare.py` và `data/loader.py`.
- [ ] Khảo sát ≈ 5.000 bài (phân bố độ dài, tỷ lệ bài < 2.000 ký tự, tần suất ký tự) — lưu bảng/biểu đồ
- [ ] NFC, gộp khoảng trắng, bỏ bài < 2.000 ký tự, **giữ chữ hoa**
- [ ] Chia theo bài viết, seed cố định: test ≈ 1M, val ≈ 1M, train ≈ 50M ký tự
- [ ] Từ vựng từ train phủ ≥ 99,9% (≈ 200–300 ký tự), id 0 `<unk>`, id 1 `<doc>`
- [ ] `train/val/test.bin` uint16, `vocab.json`, `meta.json`, `sample.bin` lên Kaggle Dataset `mamba-btl-data`
- [ ] `data/loader.py`: `WikiCharData` đọc bằng `np.memmap`, `get_batch(split, batch_size, seq_len, step, seed)` xác định
- [ ] `tests/test_loader.py`: encode/decode, `<unk>` < 0,1% trên val, không trùng bài giữa các tập, cùng seed + step -> cùng batch
- [ ] 1 trang ghi chú số liệu dữ liệu (cho chương Dữ liệu)
- [ ] Nếu kịp: tập mô phỏng gõ phím ≈ 2.000 vị trí trong test (định dạng thống nhất với ĐG)

### #3 [TF] TransformerLM
**Nhánh:** `tf/transformer-lm` · Dùng `sample.bin` (N2) và `train.py` (N3)
Đặc tả trong docstring `models/transformer_lm.py`.
- [ ] Kế thừa `models.base.BaseCharLM`; 4 block pre-norm RMSNorm, attention nhân quả SDPA `is_causal=True`, RoPE, MLP 4×
- [ ] `pytest tests/test_models.py`: shape + tính nhân quả pass
- [ ] Cân tham số với MB: ≈ 8M, lệch ≤ 5% (`test_param_balance` pass)
- [ ] L = 4.096 trên T4, ép `EFFICIENT_ATTENTION`, batch 1/2/4 -> bảng bộ nhớ đỉnh (OOM thì ghi nhận)
- [ ] 500 bước fp16 trên dữ liệu thật bằng `train.py` — loss giảm, không NaN (dán log)
- [ ] Nếu kịp: phác thảo KV cache cho tuần 2

### #4 [MB] Môi trường mamba-ssm + MambaLM
**Nhánh:** `mb/env`, `mb/mamba-lm` · **Mốc: N4 (23/11) wheel trên Kaggle Dataset hoặc chốt backend torch**
Đặc tả trong docstring `models/mamba_lm.py`.
- [ ] Ghi môi trường Kaggle (`nvidia-smi`, torch/CUDA/Python) vào `install_notes.txt`
- [ ] Cài mamba-ssm theo thứ tự ưu tiên (pip -> wheel Releases -> cài lại torch khớp); smoke test so với `selective_scan_ref`
- [ ] `*.whl` + `install_notes.txt` lên Kaggle Dataset `mamba-btl-wheels`; notebook setup cài lại được trong vài giây
- [ ] `models/mamba_lm.py`: 8 block (RMSNorm + Mamba mixer, residual, `residual_in_fp32`), cờ `backend: cuda | torch`
- [ ] Cân tham số với TF (≈ 8M, lệch ≤ 5%); `pytest tests/test_models.py` pass
- [ ] 500 bước fp16 và fp32 trên dữ liệu thật -> biểu đồ 2 đường loss
- [ ] Ghi `mamba-ssm` (và `mamba.py` nếu dùng) vào `docs/references.md`

### #5 [ĐG] train.py + checkpoint/resume + bench.py + đặc tả đánh giá
**Nhánh:** `dg/train`, `dg/bench` · **Mốc: N3 (22/11) train.py chạy được trên sample.bin**
Khung `train.py` đã có (config, vòng lặp, AdamW + cosine/warmup, fp16 + GradScaler, BPC, batch theo ký tự, log CSV). ĐG nhận sở hữu và hoàn thiện các TODO trong docstring `train.py`:
- [ ] Checkpoint dict đủ khóa (model, optimizer, scheduler, scaler, step, rng_state, config), ghi file tạm rồi đổi tên, mỗi 15–20 phút
- [ ] Tự tìm `ckpt_latest.pt` (cả trong `/kaggle/input`) và resume
- [ ] `torch.isfinite(loss)`: NaN -> bỏ bước, ghi log; lặp nhiều lần -> dừng
- [ ] Test resume (N5): 300 bước, ngắt ~150, resume -> đường loss liền mạch (dán biểu đồ)
- [ ] `bench.py` theo docstring (trung vị, OOM -> "OOM", ghi tên GPU)
- [ ] `docs/eval_spec.md` (1 trang): BPC, Top-1/Top-3 cho (a)/(b), KSR, cách đo độ trễ, quy ước chữ hoa

### #6 [TN] Hạ tầng, review, tích hợp
- [x] Cấu trúc repo theo mục 3.1, `README.txt`, `base.yaml`, `notebooks/00_setup.ipynb`, `models/base.py`, `tests/test_models.py`
- [ ] Bật bảo vệ nhánh `main` (bắt buộc PR); mời 4 thành viên
- [ ] Kaggle Datasets `mamba-btl-data`, `mamba-btl-wheels` (private, 4 thành viên quyền Editor)
- [ ] 2 Google Sheet: bảng lượt chạy, nhật ký đóng góp
- [ ] Review PR trong 24h, ưu tiên: loader (DL) -> train.py (ĐG) -> 2 model
- [ ] Check-in N4: chốt backend Mamba, ghi vào `docs/decisions.md` và README
- [ ] Nhờ 2 thành viên chạy `00_setup.ipynb` trên tài khoản của họ
- [ ] N6: chạy `train.py` cả 2 model trên dữ liệu thật, L = 256, ~300 bước; giao lỗi ghép nối
- [ ] N7: checklist tuần 1, phân lượt chạy tuần 2
