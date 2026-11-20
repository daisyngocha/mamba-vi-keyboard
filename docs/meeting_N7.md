# Họp review cuối tuần 1 (N7 · 26/11/2026 · ≈ 60 phút)

Chủ trì: TN (Lê Thị Ngọc Hà). Theo *Hướng dẫn Tuần 1* mục 9. Tài liệu đi kèm: `docs/decisions.md`, `logs/bench/README.md`, Google Sheet *mamba-vi-keyboard — Theo dõi lượt chạy & đóng góp*.

**Chuẩn bị trước họp (mỗi người):** chạy `notebooks/00_setup.ipynb` trên tài khoản Kaggle của mình (GPU T4, Environment *Pin to original environment (2026-11-19)*, Add Input 2 Dataset), đọc phần mình phải giải thích chéo ở mục 2, ghi giờ đã làm vào tab *Nhật ký đóng góp*.

## Chương trình

| # | Nội dung | Phút | Người dẫn |
|---|---|---|---|
| 1 | Kiểm tra checklist tuần 1, demo chạy được | 10 | TN |
| 2 | Giải thích chéo pipeline | 15 | Cả nhóm |
| 3 | Kết quả sơ bộ RQ2 | 5 | TN |
| 4 | Thông báo quyết định đã chốt (Q6–Q9) | 5 | TN |
| 5 | Phân lượt chạy tuần 2 | 10 | TN, MB, TF |
| 6 | Việc tuần 2 theo vai trò | 5 | TN |
| 7 | Cập nhật nhật ký đóng góp, chốt biên bản | 5 | TN |

## 1. Checklist tuần 1

| Tiêu chí (Hướng dẫn mục 1) | Trạng thái | Bằng chứng |
|---|---|---|
| Cài mamba-ssm và qua smoke test, hoặc chốt backend PyTorch | Đạt | Q7; `selective_scan_fn` lệch `selective_scan_ref` 6,5e-05; wheel ở Dataset `mamba-btl-wheels` |
| train/val/test.bin và vocab.json trên Kaggle Dataset `mamba-btl-data` | Đạt | vocab 275; train 50,0M · val 1,0M · test 1,0M ký tự; `<unk>` val 0,058% |
| 2 model chạy 500 bước fp16 trên dữ liệu thật, không NaN, loss giảm | Đạt | Mamba val BPC 2,030 · Transformer 2,336 sau 500 bước |
| Transformer chạy được ở L = 4.096 (có số đo bộ nhớ) | Đạt | B = 1/2/4/8: 453 / 786 / 1.424 / 2.732 MB, không OOM |
| Ngắt phiên Kaggle và resume đúng từ checkpoint | Đạt | issue #5: loss chênh 0,0 so với lượt không ngắt |
| ≥ 2 thành viên khác chạy được notebook setup | _điền tại họp_ | tên + ảnh output cell 4 |
| Code DL, MB, TF, ĐG merge vào main qua review | Đạt | PR #10–#19 |

Mỗi người mở notebook của mình và cho cả nhóm xem cell 4 in `mamba-ssm OK` và `SMOKE TEST OK`.

## 2. Giải thích chéo (2–3 phút mỗi người)

Mỗi người giải thích phần **của người khác**, bằng lời của mình, trả lời được 1 câu hỏi vặn. Tuần 1 TN viết code thay cả nhóm, nên phần này là lúc từng người nắm lại phần mình sở hữu từ tuần 2.

| Người trình bày | Giải thích phần | Câu hỏi vặn gợi ý |
|---|---|---|
| Thu Hằng (DL) | Mamba: `models/mamba_lm.py`, backend cuda/torch, khởi tạo A_log, D | Vì sao A_log và D không bị weight decay? |
| Tiến Dũng (MB) | Transformer: RoPE, SDPA `is_causal`, test nhân quả | Vì sao không truyền mask tự tạo vào SDPA? |
| Duyên Hà (TF) | `train.py`: batch theo ký tự, checkpoint, resume | Vì sao resume không cần lưu trạng thái dataloader? |
| Thanh Hằng (ĐG) | Dữ liệu: `prepare.py`, lọc bài < 2.000 ký tự, chia theo bài, từ vựng 99,9% | Vì sao chia theo bài viết chứ không theo ký tự? |
| Ngọc Hà (TN) | `bench.py` và `docs/eval_spec.md`: Top-k, KSR, đo độ trễ | KSR tính thế nào khi gợi ý đúng ở phím đầu tiên? |

## 3. Kết quả sơ bộ RQ2 (`logs/bench/README.md`)

Tesla T4, fp16, 32.768 ký tự/bước, trung vị 3 lần chạy:

| L | Transformer (ký tự/s) | Mamba (ký tự/s) | Mamba / Transformer |
|---|---|---|---|
| 256 | 138k | 66k | 0,47 |
| 1.024 | 120k | 74k | 0,62 |
| 4.096 | 81k | 80k | 0,99 |
| 8.192 * | 59k | 84k | 1,42 |
| 16.384 * | 34k | 84k | 2,45 |

\* ngoài thiết kế, thí nghiệm khám phá. Bộ nhớ huấn luyện ≈ 2,7 GB cho cả 2 model ở mọi L (attention memory-efficient). Điểm giao ≈ L = 4.096. Khác biệt bộ nhớ sẽ phải thể hiện ở RQ3 (KV cache so với trạng thái cố định).

## 4. Quyết định cần chốt

| # | Đề xuất | Căn cứ | Chốt |
|---|---|---|---|
| Q6 | `base.yaml`: AdamW lr 1e-3, β (0,9; 0,95), wd 0,1, warmup 2% + cosine về 10%, batch 32.768 ký tự, **train_chars 200M (≈ 4 epoch)**, eval mỗi 250 bước, checkpoint mỗi 15 phút | 500 bước ổn định, không NaN; ước tính 0,7–1,2 giờ GPU mỗi lượt | ☑ 20/11 |
| Q7 | Backend Mamba `cuda` (mamba-ssm 2.3.2.post1), Environment Kaggle ghim 2026-11-19 | smoke test + test so với bản PyTorch tham chiếu | ☑ 20/11 |
| Q8 | **fp16**; d_model Transformer 384 (6 × 64), Mamba 368 → 7,19M / 7,20M tham số (lệch 0,18%); **L tối đa 4.096** | fp16 = fp32 tới 4 chữ số BPC, nhanh gấp 2; L = 4.096 B = 8 chỉ dùng 2,7 GB | ☑ 20/11 |
| Q9 | Đặc tả đánh giá `docs/eval_spec.md`: Top-k khớp chính xác cả hoa/thường (+ chỉ số phụ không phân biệt); KSR chạm chọn = 1 phím, không tự thêm khoảng trắng; beam 5, ≤ 10 ký tự | Q2 giữ chữ hoa | ☑ 20/11 |

TN đã chốt cả 4 quyết định theo đề xuất ngày 20/11 (ghi trong `docs/decisions.md`). Tại họp chỉ cần thông báo lại; nếu cả nhóm muốn đổi, ghi thay đổi vào `decisions.md`.

## 5. Lượt chạy tuần 2 (27/11–03/12)

Thứ tự ưu tiên MVP theo tài liệu: P1 (L = 256, 1.024, seed 0) → P2 (L = 4.096) → P3 (seed 1, 2 tại L = 1.024). Ghi từng lượt vào tab *Lượt chạy* của Google Sheet trước khi bấm chạy.

| Ưu tiên | Lượt | Người chạy · tài khoản | Giờ GPU ước tính |
|---|---|---|---|
| P1 | `mamba_L256_s0`, `mamba_L1024_s0` | Tiến Dũng (MB) | 1,2 + 1,1 |
| P1 | `transformer_L256_s0`, `transformer_L1024_s0` | Duyên Hà (TF) | 0,7 + 0,75 |
| P2 | `mamba_L4096_s0` | Tiến Dũng (MB) | 1,0 |
| P2 | `transformer_L4096_s0` | Duyên Hà (TF) | 1,0 |
| P3 | `mamba_L1024_s1`, `mamba_L1024_s2` | Thu Hằng (DL), mượn tài khoản | 1,1 + 1,1 |
| P3 | `transformer_L1024_s1`, `transformer_L1024_s2` | Thanh Hằng (ĐG), mượn tài khoản | 0,75 + 0,75 |
| | **Tổng** | | **≈ 9,5 giờ**, mỗi tài khoản ≤ 3,3 giờ |

Ước tính = 200M ký tự ÷ (tốc độ `bench.py` × 0,8 cho đọc dữ liệu, eval, log) + khoảng 10 phút cài đặt. Một lượt dùng gọn trong giới hạn 12 giờ của Save & Run All.

**Cách chạy một lượt:** mở `notebooks/train_run.ipynb` trên Kaggle (File → Import Notebook → GitHub), sửa 3 dòng `MODEL`, `SEQ_LEN`, `SEED` ở cell đầu, Add Input 2 Dataset, bấm **Save Version → Save & Run All**. Lượt bị ngắt thì Add Input output của version đó và chạy lại đúng notebook: `train.py` tự resume. Xong lượt: điền Giờ GPU, Val BPC, ký tự/s từ `_summary.json` vào Sheet; chép `logs/<run_name>.csv` và `_summary.json` vào `logs/` của repo qua PR.

## 6. Việc tuần 2 theo vai trò (Mô tả bài toán mục 5, Hướng dẫn mục 1)

Từ tuần 2 mỗi người tự làm và tự commit trên nhánh `<mã>/<việc>` của mình; TN chỉ review và tích hợp.

| Vai trò | Việc tuần 2 | Đầu ra |
|---|---|---|
| MB · Tiến Dũng | Chạy các lượt Mamba P1–P2; viết `MambaLM.step(token, state)` sinh từng bước với `conv_state`, `ssm_state` của mamba-ssm; test: logits từ `step()` = `forward` trên cả chuỗi (sai số ≤ 1e-3) | checkpoint + log; PR `mb/step` |
| TF · Duyên Hà | Chạy các lượt Transformer P1–P2; viết `TransformerLM.step()` với KV cache (RoPE theo vị trí tuyệt đối); cùng test như MB | checkpoint + log; PR `tf/kv-cache` |
| ĐG · Thanh Hằng | `evaluate/suggest.py` (beam search theo `eval_spec.md` mục 2) và `evaluate/ksr.py` (Top-1/Top-3, KSR); chạy thử trên checkpoint 500 bước; lượt P3 Transformer | PR `dg/suggest`, `dg/ksr` |
| DL · Thu Hằng | Trang ghi chú số liệu dữ liệu (chương Dữ liệu); hàm phân loại lỗi dấu thanh / dấu chữ / viết hoa / chữ gốc (`eval_spec.md` mục 6); lượt P3 Mamba | `docs/data_notes.md`; PR `dl/error-analysis` |
| TN · Ngọc Hà | Review PR trong 24 giờ; `notebooks/figures.ipynb` (bảng và biểu đồ tự sinh từ `logs/`); theo dõi quota | PR `tn/figures` |

## 7. Biên bản (điền tại họp)

- Có mặt: ☐ Ngọc Hà ☐ Thu Hằng ☐ Tiến Dũng ☐ Duyên Hà ☐ Thanh Hằng
- Quyết định: Q6–Q9 đã chốt 20/11 — thay đổi tại họp (nếu có):
- Thành viên đã chạy được `00_setup` trên tài khoản riêng:
- Vướng mắc và người xử lý:
- Hạn tiếp theo: checkpoint P1 trước 30/11; `step()` của cả 2 model trước 01/12; `suggest` chạy được trước 03/12.
