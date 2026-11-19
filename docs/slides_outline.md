# Dàn ý slide — Buổi báo cáo 2: Kết quả thực nghiệm Mamba vs Transformer trên bài toán gợi ý từ bàn phím tiếng Việt

Phần lý thuyết Mamba đã trình bày ở buổi 1, nên buổi này chỉ nhắc lại 1 slide và dồn thời lượng cho thực nghiệm.
Thời lượng: **[CHỐT] 30 phút trình bày + 15 phút Q&A** (giả định, chỉnh theo yêu cầu của thầy) · Template: HUST · Người trình bày: Lê Thị Ngọc Hà
Mục tiêu: ~20 slide chính + 5–6 slide dự phòng (appendix) cho Q&A.

Nguyên tắc cho mọi slide kết quả:
- Tiêu đề slide là **kết luận** (vd "Mamba dùng bộ nhớ tuyến tính, Transformer OOM ở L = 4096"), không phải tên biểu đồ.
- Mọi bảng và biểu đồ sinh tự động từ `results/` bằng `notebooks/make_figures.ipynb`, không nhập số tay.
- Ghi rõ GPU, số seed, và sai số (mean ± std) nếu chạy nhiều seed.

## Tổng quan
| # | Phần | Phút | Slide | Soạn nháp | Nguồn số liệu / sơ đồ |
|---|---|---|---|---|---|
| 1 | Mở đầu + nhắc lại Mamba trong 1 slide | 2 | 2 | Ngọc Hà | `Mamba_Architecture` |
| 2 | Bài toán và câu hỏi thực nghiệm | 3 | 2 | Ngọc Hà | — |
| 3 | Dữ liệu | 2 | 2 | Thu Hằng | `data/processed/stats.json` |
| 4 | Cài đặt Mamba và kiểm chứng | 3 | 2 | Tiến Dũng | `Mamba_Block_and_Scan`, log unit test |
| 5 | Thiết lập thí nghiệm | 3 | 2 | Duyên Hà | `configs/*.yaml`, `summary.json` |
| 6 | Kết quả E1–E4 | 12 | 7 | Duyên Hà (E2) · Thanh Hằng (E1, E3, E4) | `results/` |
| 7 | Phân tích lỗi + demo | 3 | 2 | Thu Hằng | `eval_test.json`, video demo |
| 8 | Thảo luận, hạn chế, kết luận | 2 | 1 | Ngọc Hà | — |
| | **Tổng** | **30** | **20** | | |
| A | Appendix (không trình bày, dùng khi Q&A) | — | 5–6 | Theo từng người | |

## Nội dung chi tiết

**1. Mở đầu (2 slide)**
- 1.1 Tên đề tài, thành viên, agenda.
- 1.2 Nhắc lại Mamba trong 1 slide: SSM có chọn lọc (Δ, B, C phụ thuộc input), train song song bằng scan, chi phí tuyến tính theo L. Dẫn vào giả thuyết cần kiểm chứng: *chất lượng ngang Transformer, hiệu quả hơn ở chuỗi dài*.

**2. Bài toán và câu hỏi thực nghiệm (2 slide)**
- 2.1 Bài toán: gõ ngữ cảnh + prefix → top-3 từ gợi ý. Minh hoạ 1 ví dụ (`"hôm nay trời rất n"` → `nóng / nắng / nhiều`). Vì sao chọn bài toán này: cần chạy nhanh trên thiết bị, ngữ cảnh dài, phù hợp mô hình char-level nhỏ.
- 2.2 Bốn câu hỏi nghiên cứu, mỗi câu gắn với 1 thí nghiệm:

| RQ | Câu hỏi | Thí nghiệm |
|---|---|---|
| RQ1 | Bản Mamba tự cài đặt có thể hiện đúng cơ chế selection không? | E1 Selective Copying |
| RQ2 | Chất lượng gợi ý từ của Mamba so với Transformer cùng cỡ tham số? | E2 BPC, Top-k, KSR |
| RQ3 | Chi phí train (tốc độ, bộ nhớ) khác nhau thế nào khi tăng độ dài chuỗi? | E3 scaling theo L |
| RQ4 | Độ trễ suy luận có đủ cho bàn phím (< 50 ms/gợi ý)? | E4 latency |

**3. Dữ liệu (2 slide)**
- 3.1 Nguồn (Wikipedia tiếng Việt), quy mô sau làm sạch, pipeline xử lý: NFC → chữ thường → lọc → vocab ký tự → chia theo bài viết.
- 3.2 Thống kê: số ký tự train/val/test, kích thước vocab, tỉ lệ `<unk>`, phân bố độ dài từ; cách tạo 2.000 mẫu đánh giá gợi ý từ (k = 0, 1, 2).

**4. Cài đặt Mamba và kiểm chứng (2 slide)**
- 4.1 Sơ đồ module tự cài đặt (`mamba_pt`): in_proj → conv1d → x_proj (Δ, B, C) → selective scan → gate → out_proj. Chỉ rõ đoạn code ứng với selection.
- 4.2 Bằng chứng cài đặt đúng:
  - Causality test pass.
  - Overfit 1 batch (đường loss).
  - So khớp output với `mamba-ssm` (sai số tối đa), nếu cài được.
  - Tốc độ scan tuần tự so với kernel CUDA. Đây là lý do E3 cần tách 2 bản cài đặt.

**5. Thiết lập thí nghiệm (2 slide)**
- 5.1 Bảng cấu hình 2 mô hình: d_model, số lớp, **số tham số thực tế**, d_state, n_head…
- 5.2 Giao thức công bằng: cùng dữ liệu, vocab, seq_len, batch, optimizer, lịch learning rate, số bước; quét lr cho cả 2; số seed; phần cứng (Kaggle T4/P100), AMP.

**6. Kết quả (7 slide, phần trọng tâm)**
- 6.1 **E1 Selective Copying**: bảng accuracy Mamba vs Transformer (vs S4/LTI nếu kịp, tức Mamba tắt selection). Kết luận cho RQ1.
- 6.2 **E2 Đường học**: val BPC theo số bước và theo thời gian thực (2 subplot). Mô hình nào hội tụ nhanh hơn tính theo wall-clock?
- 6.3 **E2 Chất lượng tác vụ**: bảng test BPC, Top-1/Top-3 tại k = 0, 1, 2, KSR. Kết luận cho RQ2.
- 6.4 **E3 Tốc độ train**: tokens/s ở seq_len 256 (bar chart) cho Transformer, `mamba_pt`, `mamba_ssm`.
- 6.5 **E3 Scaling theo độ dài chuỗi**: thời gian fwd+bwd và peak memory theo L ∈ {256 … 4096}, trục log; đánh dấu điểm OOM. **Đây là biểu đồ quan trọng nhất buổi báo cáo.**
- 6.6 **E4 Latency suy luận**: ms/gợi ý theo độ dài context. Transformer có/không KV cache, Mamba dùng chế độ recurrent. Kết luận cho RQ4.
- 6.7 **Tổng hợp**: 1 bảng trả lời RQ1–RQ4 (✓ / ✗ / một phần), so với kỳ vọng từ paper.

**7. Phân tích lỗi + demo (2 slide)**
- 7.1 5–6 ví dụ gợi ý đúng/sai của cả 2 mô hình. Nhóm lỗi điển hình: từ hiếm, tên riêng, ngữ cảnh xa, sai dấu thanh.
- 7.2 Demo: video 30–45 giây gõ thử (notebook/Gradio), chạy song song 2 mô hình. Mang theo bản video phòng khi demo trực tiếp lỗi.

**8. Kết luận (1 slide)**
- Kết quả xác nhận được và chưa xác nhận được so với paper.
- Hạn chế: mô hình nhỏ (~3M tham số), dữ liệu Wikipedia khác văn gõ hằng ngày, chưa hỗ trợ Telex, ít seed.
- Hướng mở rộng: dữ liệu hội thoại/tin nhắn, Telex, Mamba-2 hoặc hybrid, lượng tử hoá để chạy trên điện thoại.

**Appendix (dự phòng cho Q&A)**
- A1 Bảng siêu tham số đầy đủ + kết quả quét learning rate.
- A2 `mamba_pt` vs `mamba_ssm`: tốc độ, độ lệch output.
- A3 Kết quả từng seed.
- A4 Ablation nếu kịp: d_state ∈ {4, 16}; tắt selection (Δ, B, C cố định).
- A5 Chi tiết cách tính KSR và Top-k.
- A6 Chi phí tính toán: tổng giờ GPU đã dùng.

## Mốc thời gian **[CHỐT theo ngày báo cáo]**
| Mốc | Hạn |
|---|---|
| Khung slide với biểu đồ giữ chỗ (placeholder) + `make_figures.ipynb` chạy với kết quả smoke test | 06/12 |
| Số liệu cuối E1–E4, chèn vào slide | 12/12 |
| Tập dượt lần 1 (bấm giờ, cả nhóm hỏi vặn) | 14/12 |
| Tập dượt lần 2 + chốt appendix | 16/12 |

## Chuẩn bị Q&A (mỗi người soạn câu trả lời ≤ 5 dòng cho câu được giao)
| Câu hỏi dự kiến | Người soạn |
|---|---|
| So sánh có công bằng không (số tham số, learning rate, số bước, phần cứng)? | Duyên Hà |
| Làm sao biết bản Mamba tự cài đặt là đúng? | Tiến Dũng |
| Vì sao `mamba_pt` chậm hơn Transformer dù lý thuyết là tuyến tính? | Tiến Dũng |
| Nếu Transformer thắng về BPC thì kết luận gì? | Ngọc Hà |
| Kết quả có ổn định qua các seed không? Sai số bao nhiêu? | Thanh Hằng |
| Vì sao chọn char-level, không dùng BPE/syllable tokenizer? | Thu Hằng |
| Tập đánh giá gợi ý từ được tạo thế nào, có rò rỉ dữ liệu không? | Thu Hằng |
| KSR tính thế nào, có phản ánh trải nghiệm người dùng thật không? | Thanh Hằng |
| Có triển khai lên điện thoại được không (kích thước, latency)? | Thanh Hằng |
| Nếu có thêm thời gian, nhóm sẽ làm gì tiếp? | Ngọc Hà |
