# Scope — BTL "Nền tảng GenAI": Mamba vs Transformer cho gợi ý từ bàn phím tiếng Việt

Phiên bản: **v1.0** (chốt 19/11/2026) · Owner: Lê Thị Ngọc Hà (trưởng nhóm)
Các quyết định D1–D7 đã chốt ngày 19/11/2026 — xem decision log trong `docs/kickoff_agenda.md`. Thay đổi scope sau ngày này phải qua PR và được trưởng nhóm duyệt.

## 1. Yêu cầu của đề
- Buổi 1 (đã xong): trình bày lý thuyết paper Mamba (Gu & Dao).
- **Buổi 2 (BTL này)**: báo cáo kết quả thực nghiệm trên bài toán thực tế — cài đặt Mamba, huấn luyện, đánh giá và so sánh với Transformer về **độ chính xác, tốc độ huấn luyện, bộ nhớ**.
    Dàn ý: `docs/slides_outline.md`.
- Ngày + thời lượng buổi 2: **26/12/2026, 30 phút trình bày + 15 phút Q&A** (D1).

## 2. Bài toán ứng dụng
**Gợi ý từ cho bàn phím tiếng Việt** bằng language model mức ký tự (char-level), train từ đầu.

- Input: ngữ cảnh văn bản đã gõ + prefix của từ đang gõ (có thể rỗng).
  Ví dụ: `"hôm nay trời rất n"` → gợi ý top-3: `nóng`, `nắng`, `nhiều`.
- Output: top-3 từ hoàn chỉnh.
- Cách sinh: mô hình dự đoán ký tự kế tiếp; beam search (beam = 5) sinh tiếp ký tự cho tới khi gặp dấu cách/dấu câu → mỗi beam là 1 từ ứng viên; gộp trùng, xếp theo log-prob.

**Ngoài phạm vi**: bộ gõ Telex/VNI (giả định input đã là ký tự Unicode hoàn chỉnh), sửa lỗi chính tả, cá nhân hoá theo người dùng, triển khai lên điện thoại thật.

## 3. Dữ liệu
- Nguồn: Wikipedia tiếng Việt — HF `wikimedia/wikipedia`, config `20231101.vi`. Tuần 1 chỉ dùng Wikipedia; cân nhắc bổ sung corpus tin tức (VNTC) ở tuần 2 nếu còn thời gian (D2).
- Quy mô: tập con ~50–100 MB text sau làm sạch (đủ cho mô hình ~3–5M tham số, train được trên GPU Kaggle trong vài giờ).
- Tiền xử lý (bắt buộc):
  - Chuẩn hoá Unicode **NFC**.
  - Chuyển về chữ thường (D3).
  - Bỏ markup, bảng, dòng quá ngắn (< 20 ký tự); gộp khoảng trắng.
  - Vocab ký tự: giữ ký tự xuất hiện ≥ 50 lần, còn lại → `<unk>`; dự kiến 100–150 ký tự.
- Chia tập **theo bài viết** (không theo câu) để tránh rò rỉ: train 90% / val 5% / test 5%, seed cố định = 42.
- Tập đánh giá gợi ý từ: 2.000 mẫu `(context, prefix, target_word)` lấy từ test set, context dài 64–256 ký tự, prefix k ∈ {0, 1, 2} ký tự.

## 4. Mô hình
| | Transformer (baseline) | Mamba |
|---|---|---|
| Kiểu | GPT decoder-only, pre-LN, causal | Mamba block (S6 + conv1d + gating), pre-norm RMSNorm |
| d_model | 256 | 256 |
| Số lớp | 4 | 8 (≈ 2 Mamba block / 1 Transformer layer về tham số) |
| Khác | 4 heads, FFN 4×, learned pos-emb | d_state 16, d_conv 4, expand 2 |
| Tham số (ước tính) | ~3.2M | ~3.5M |

- Sau khi cài xong, đếm tham số thật và chỉnh `n_layer` để chênh lệch < 10%.
- Mamba có 2 bản cài đặt:
  - `mamba_pt` — PyTorch thuần, selective scan tuần tự. **Đây là phần "tự cài đặt" để chấm**, dùng để giải thích trên slide.
  - `mamba_ssm` — package chính thức (CUDA kernel). Dùng cho benchmark tốc độ nếu cài được trên Kaggle. Kiểm tra 2 bản cho output khớp nhau (sai số < 1e-3) khi dùng chung trọng số.

## 5. Giao thức so sánh (fairness)
Giống nhau cho cả 2 mô hình: dữ liệu, vocab, seq_len train = 256, batch 64, AdamW (lr 1e-3 cho Mamba; quét lr {3e-4, 1e-3} cho Transformer — D4), weight decay 0.1, warmup 500 bước + cosine, grad clip 1.0, AMP bf16/fp16, **cùng ngân sách số bước: 20k bước** (D5). Mỗi cấu hình chạy 3 seed nếu đủ quota GPU (tối thiểu 1 seed).

## 6. Metric
1. **Chất lượng mô hình**: val/test loss, bits-per-char (BPC).
2. **Chất lượng tác vụ**: Top-1 / Top-3 word accuracy theo k ∈ {0,1,2}; Keystroke Savings Rate (KSR).
3. **Hiệu năng**:
   - Train throughput (tokens/s), thời gian tới khi đạt cùng 1 mức val loss.
   - Peak GPU memory (`torch.cuda.max_memory_allocated`).
   - Latency 1 lần gợi ý (ms, batch 1).
   - **Scaling theo độ dài chuỗi**: thời gian + bộ nhớ forward/backward với L ∈ {256, 512, 1024, 2048, 4096} — kết quả trung tâm của phần so sánh.

## 7. Danh sách thí nghiệm
| ID | Nội dung | Mục đích trên slide |
|---|---|---|
| E1 | Selective Copying (tái hiện nhỏ Table 1 của paper) | Chứng minh cơ chế selection |
| E2 | Train LM char-level trên dữ liệu tiếng Việt | So sánh BPC + Top-k accuracy |
| E3 | Benchmark thời gian/bộ nhớ theo độ dài chuỗi | Minh hoạ linear vs quadratic |
| E4 | Demo gợi ý từ (notebook/Gradio) | Kết quả ứng dụng |

## 8. Timeline
| Tuần | Thời gian | Mục tiêu |
|---|---|---|
| 1 | 20–26/11 | Repo, dữ liệu, 2 model chạy end-to-end, sanity test, dàn ý slide |
| 2 | 27/11–03/12 | Train đầy đủ E2, chạy E1, bộ eval hoàn chỉnh |
| 3 | 04–10/12 | E3 benchmark, demo E4, bản nháp slide đầy đủ |
| 4 | 11–17/12 | Hoàn thiện slide, tập dượt 2 lần, chuẩn bị Q&A — **hoàn thành toàn bộ trước 17/12** |
| — | 18–25/12 | Dự phòng (không lên kế hoạch việc mới) |
| — | **26/12** | **Báo cáo buổi 2** |

## 9. Phân công tuần 1 (D6)
| Người | Phạm vi |
|---|---|
| Lê Thị Ngọc Hà (lead) | Scope, repo, config, review PR, dàn ý slide, điều phối |
| Nguyễn Tiến Dũng | `src/models/mamba_pt.py`, thử `mamba_ssm` |
| Đặng Duyên Hà | `src/models/transformer.py`, hoàn thiện `src/train.py` |
| Ngô Thị Thu Hằng | `data/prepare_data.py`, `src/dataset.py`, tập eval gợi ý từ |
| Cao Thanh Hằng | `src/evaluate.py`, `src/benchmark.py`, dữ liệu Selective Copying |

## 10. Quy ước làm việc
- Git: mỗi người làm trên nhánh `feat/<tên-task>`, merge vào `main` qua PR, lead review trong 24h. Không commit dữ liệu thô/checkpoint (đã có `.gitignore`).
- Mọi thí nghiệm chạy qua `python -m src.train --config ...`, kết quả ghi vào `results/<run_name>/` (config + metrics.csv) → bảng kết quả trên slide được sinh từ đây.
- Daily update (nhóm chat, trước 22h): *Hôm nay làm gì · Bị chặn ở đâu · Mai làm gì*.
- Sync: Thứ 2 & Thứ 4 hằng tuần, 21:00, 30–45 phút.
- Quota GPU Kaggle (~30h/tuần/tài khoản): run dài > 1h phải báo trước trong nhóm.

## 11. Rủi ro
| Rủi ro | Phương án |
|---|---|
| Không cài được `mamba_ssm` | Dùng `mamba_pt`; slide nói rõ benchmark tốc độ chưa có kernel tối ưu, giải thích vì sao |
| Thiếu GPU | Giảm dữ liệu/bước train; phân quota theo người |
| Thành viên bận môn khác | Mỗi task 1 owner + deadline; CN chưa có PR → lead chia lại ở sync Thứ 2 |
| Transformer thắng về BPC ở seq_len ngắn | Đây vẫn là kết quả hợp lệ; nhấn mạnh E3 + phân tích trade-off |
