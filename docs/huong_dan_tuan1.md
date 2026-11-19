# Hướng dẫn công việc tuần 1 theo vai trò (20–26/11/2026)

Phiên bản: **v1.1** (cập nhật 19/11/2026, sau khi chốt D1–D7) · Owner: Lê Thị Ngọc Hà
Ký hiệu: 🔄 = thay đổi so với bản hướng dẫn đầu · 🆕 = nội dung mới bổ sung · ✅ = đã xong
Tài liệu liên quan: [`scope.md`](scope.md) (v1.0) · [`github_issues_week1.md`](github_issues_week1.md) · [`slides_outline.md`](slides_outline.md)

## 1. Bối cảnh
- 🔄 Buổi 1 (lý thuyết Mamba) **đã xong**. BTL còn lại là **buổi 2: chỉ báo cáo kết quả thực nghiệm**, ngày **26/12/2026**, 30 phút + 15 phút Q&A (D1). **Không ai phải soạn slide lý thuyết.**
- 🔄 Timeline 4 tuần, **hoàn thành toàn bộ trước 17/12**; 18–25/12 chỉ để dự phòng.
- Mục tiêu tuần 1: chốt phạm vi và có pipeline chạy được từ đầu đến cuối cho cả 2 mô hình. Chưa cần tối ưu kết quả.

## 2. Mục tiêu cuối tuần 1 (Definition of Done, tối 26/11)
1. ✅ Scope đã chốt — 🔄 `scope.md` v1.0, D1–D7 chốt ngày 19/11.
2. ✅ Repo GitHub có cấu trúc chuẩn, chạy được trên GPU Kaggle — 🆕 đã xác minh trên **Tesla T4** (smoke test OK, chi tiết issue #6).
3. Pipeline dữ liệu tiếng Việt xong: đã làm sạch, tách train/val/test, có vocab ký tự và DataLoader.
4. Transformer baseline train hết một vòng, có log loss, thời gian và bộ nhớ.
5. Mamba đã cài xong và qua sanity test: overfit được 1 batch, giải được Selective Copying cỡ nhỏ.
6. 🔄 Dàn ý slide **buổi 2 (30 phút, chỉ thực nghiệm)** v1.0, mỗi phần đã có người phụ trách — thay cho dàn ý 60 phút có phần lý thuyết.

## 3. 🆕 Các quyết định đã chốt áp dụng cho tuần 1
| # | Quyết định | Ảnh hưởng tới tuần 1 |
|---|---|---|
| D1 | Báo cáo buổi 2 ngày 26/12, 30' + 15' Q&A | Không soạn slide lý thuyết; dàn ý theo `slides_outline.md` |
| D2 | Tuần 1 chỉ dùng Wikipedia tiếng Việt | Thu Hằng không cần xử lý corpus tin tức |
| D3 | Chuyển về chữ thường | Áp dụng trong `prepare_data.py` (`lowercase: true`) |
| D4 | Quét lr {3e-4, 1e-3} cho Transformer | Chưa chạy ở tuần 1; tuần 1 chỉ cần overfit + train ngắn |
| D5 | 20k bước, seq_len 256, batch 64 | Tuần 1 chỉ train ngắn (~1 giờ GPU/model) |
| D6 | Phân công theo scope mục 9 | Giữ nguyên người phụ trách |
| D7 | Log bằng CSV trong `results/` | Không bắt buộc W&B |

## 4. Lịch tuần
| Ngày | Việc | Ghi chú |
|---|---|---|
| 🔄 **Thứ 5 19/11**, 20:30 | Kickoff | 🔄 Dời lên sớm 1 ngày (bản đầu ghi 20/11); 🔄 phần biểu quyết đổi thành thông báo D1–D7 |
| Thứ 6 20/11 – CN 23/11 | Mỗi người làm issue của mình | Daily update trước 22h |
| 🆕 **CN 23/11, 22:00** | **Hạn mở PR** cho phần việc chính | Trước khi mở PR chạy `bash scripts/smoke_test.sh` |
| Thứ 2 24/11, 21:00 | Sync: ghép pipeline, chạy 200 bước mỗi model | |
| Thứ 3 25/11 – Thứ 4 26/11 | Sanity test, train ngắn, benchmark lần đầu | |
| Thứ 4 26/11, 21:00 | Tổng kết tuần 1, lập kế hoạch tuần 2 | Đóng milestone tuần 1 |

## 5. Việc theo vai trò
🔄 Số issue bên dưới là số **trên GitHub** (#1 là PR khởi tạo repo nên các issue bắt đầu từ #2).

### Trưởng nhóm — Lê Thị Ngọc Hà · issue #6
- ✅ Viết scope, dựng repo, config chung, chạy smoke test trên Kaggle.
- ✅ 🔄 `slides_outline.md` viết lại cho buổi 2 (chỉ thực nghiệm).
- Review mọi PR trong 24 giờ, đối chiếu với checklist trong issue.
- Tối CN 23/11: ai chưa có PR thì nhắn riêng, chuẩn bị chia lại việc ở sync Thứ 2.
- Thứ 2 24/11: merge PR đạt, chạy `src.train` 200 bước cho mỗi model trên Kaggle.
- Thứ 3–4: sanity test, train ngắn (~1 giờ GPU/model), benchmark lần đầu; tối 26/11 tổng kết.

### Dữ liệu — Ngô Thị Thu Hằng · issue #2
- Cài đặt `data/prepare_data.py`: NFC → 🔄 **chữ thường (D3)** → lọc → vocab ký tự → chia theo bài viết (train/val/test 90/5/5, seed 42).
- 🔄 Chỉ dùng **Wikipedia tiếng Việt** (D2), tập con ~50–100 MB.
- Sinh 2.000 mẫu đánh giá gợi ý từ `wordsuggest_test.jsonl`.
- 🆕 Upload `data/processed/` lên **Kaggle Dataset** (private, chia sẻ cho nhóm) để không ai phải xử lý lại.

### Transformer — Đặng Duyên Hà · issue #3
- Cài đặt `src/models/transformer.py` (GPT char-level, causal, `max_seq_len` 4096).
- Số tham số ~3–3.5M; overfit 1 batch (loss < 0.1).
- 🆕 Sửa `train.py`: `mean_tokens_per_s` **bỏ khoảng log đầu tiên** (CUDA warmup làm bước 10 chậm hơn ~10 lần trong smoke test).
- 🔄 Quét lr (D4) để sang tuần 2, tuần 1 không cần.

### Mamba — Nguyễn Tiến Dũng · issue #4
- Cài đặt `src/models/mamba_pt.py` (PyTorch thuần, selective scan tuần tự).
- Kiểm tra causality, overfit 1 batch; số tham số lệch < 10% so với Transformer.
- Thử `mamba-ssm` trên Kaggle tối đa 2 giờ; không cài được thì ghi log lỗi vào issue và dùng `mamba_pt`.
- (Không đổi so với bản đầu.)

### Đánh giá + benchmark — Cao Thanh Hằng · issue #5
- `src/evaluate.py`: BPC, Top-1/Top-3 tại k = 0, 1, 2, KSR.
- `src/benchmark.py`: thời gian + bộ nhớ theo seq_len (warmup 3 lần trước khi đo), xử lý OOM; dữ liệu Selective Copying.
- 🆕 `notebooks/make_figures.ipynb`: đọc `results/*/summary.json` + `metrics.csv`, vẽ biểu đồ cho slide buổi 2; chạy được với kết quả smoke test.

## 6. 🆕 Quy ước chạy trên Kaggle
Cell đầu notebook (chạy lại bao nhiêu lần cũng không bị clone lồng thư mục):
```python
%cd /kaggle/working
!rm -rf mamba-vi-keyboard
!git clone -q https://github.com/daisyngocha/mamba-vi-keyboard.git
%cd /kaggle/working/mamba-vi-keyboard
!pip install -q datasets pyyaml
!bash scripts/smoke_test.sh
```
- Cần test nhánh khác: thêm `!git checkout feat/...` sau dòng `%cd` thứ hai.
- Xem kết quả: đọc `results/<run>/summary.json` và `metrics.csv` bằng pandas; **không commit** `results/`, `*.pt`, `*.bin` (đã có trong `.gitignore`).
- 🆕 `main` được bảo vệ bằng ruleset: mọi thay đổi phải qua PR.

## 7. Lịch sử thay đổi
| Phiên bản | Ngày | Thay đổi |
|---|---|---|
| v1.0 | 19/11/2026 | Bản hướng dẫn đầu (theo scope v0.1) |
| v1.1 | 19/11/2026 | 🔄 Buổi 2 chỉ thực nghiệm, báo cáo 26/12, xong trước 17/12 · 🔄 DoD #6 đổi sang dàn ý buổi 2 · 🔄 Kickoff dời lên 19/11 · 🆕 Hạn PR CN 23/11 22:00 · 🔄 Số issue theo GitHub (#2–#6) · 🆕 Bảng D1–D7 · 🆕 Việc bổ sung: Kaggle Dataset (#2), warmup (#3), `make_figures.ipynb` (#5) · 🆕 Quy ước Kaggle + ruleset |
