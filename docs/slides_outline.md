# Dàn ý slide — Buổi báo cáo 2 (26/12/2026): Mamba vs Transformer cho bàn phím gợi ý từ tiếng Việt

30 phút trình bày + 15 phút Q&A · Template HUST · Chỉ trình bày thực nghiệm (lý thuyết Mamba đã xong ở buổi 1).
Mục tiêu: ~18 slide chính + 5 slide dự phòng. Theo Mô tả bài toán v1: 3 câu hỏi RQ1–RQ3, không có Selective Copying.

Nguyên tắc cho mọi slide kết quả:
- Tiêu đề slide là **kết luận**, không phải tên biểu đồ.
- Bảng/biểu đồ sinh tự động từ `logs/` bằng `notebooks/figures.ipynb`, không nhập số tay.
- Ghi rõ GPU, số seed, mean ± std tại L = 1.024 (3 seed).

| # | Phần | Phút | Slide | Người soạn | Nguồn số liệu |
|---|---|---|---|---|---|
| 1 | Mở đầu, nhắc Mamba trong 1 slide, bài toán bàn phím, RQ1–RQ3 | 4 | 3 | TN | — |
| 2 | Dữ liệu: Wikipedia vi, lọc, NFC, giữ chữ hoa, chia theo bài, từ vựng, tập mô phỏng gõ phím | 3 | 2 | DL | `meta.json`, ghi chú dữ liệu |
| 3 | Hai model và giao thức công bằng: cùng tham số (≈ 8M), cùng phần dùng chung, batch theo ký tự, cùng số ký tự huấn luyện | 4 | 3 | MB + TF | `configs/`, `num_params()` |
| 4 | RQ1 — Độ chính xác: BPC test, Top-1/Top-3 (a)/(b), KSR | 5 | 3 | ĐG | `logs/*_eval.json` |
| 5 | RQ2 — Huấn luyện theo L: ký tự/giây, thời gian, bộ nhớ đỉnh, điểm OOM | 5 | 2 | ĐG (+ MB, TF) | `logs/*.csv`, `logs/bench_*.csv` |
| 6 | RQ3 — Chạy trên thiết bị: độ trễ gợi ý và bộ nhớ theo độ dài văn bản đã gõ (256 -> 8.192), kích thước model | 4 | 2 | ĐG | `logs/latency_*.csv` |
| 7 | Khám phá: L dài có giúp gợi ý? ngoại suy độ dài; lỗi dấu thanh; (tùy chọn) CPU | 2 | 1 | DL + ĐG | — |
| 8 | Demo bàn phím (video 30–45 giây, dự phòng demo trực tiếp lỗi) | 2 | 1 | ĐG | `demo/app.py` |
| 9 | Kết luận, hạn chế (Wikipedia ≠ văn tin nhắn, ít seed), hướng mở rộng (1 slide) | 1 | 1 | TN | — |
| | **Tổng** | **30** | **18** | | |

**Dự phòng (Q&A):** bảng siêu tham số đầy đủ · kết quả từng seed · backend Mamba đã dùng và ảnh hưởng tới số đo · cách tính KSR/Top-k · tổng giờ GPU đã dùng.

## Mốc
| Mốc | Hạn |
|---|---|
| Khung slide với biểu đồ giữ chỗ + `figures.ipynb` chạy với log smoke test | 06/12 |
| Số liệu cuối, chèn vào slide | 12/12 |
| Tập dượt lần 1 (bấm giờ, cả nhóm hỏi vặn) | 14/12 |
| Tập dượt lần 2, chốt slide dự phòng — hoàn thành toàn bộ trước 17/12 | 16/12 |

## Chuẩn bị Q&A (mỗi người soạn câu trả lời ≤ 5 dòng; ai cũng phải hiểu toàn pipeline)
| Câu hỏi dự kiến | Người soạn |
|---|---|
| So sánh có công bằng không (tham số, số ký tự huấn luyện, beam search, phần cứng)? | TF |
| Đã cài Mamba thế nào, kiểm chứng đúng ra sao? Nếu dùng backend torch thì số đo tốc độ có ý nghĩa không? | MB |
| Vì sao mức ký tự, vì sao giữ chữ hoa? | DL |
| Tập mô phỏng gõ phím tạo thế nào, có rò rỉ dữ liệu không? | DL |
| KSR tính thế nào, có phản ánh trải nghiệm người dùng thật không? | ĐG |
| Kết quả có ổn định qua các seed không? | ĐG |
| Nếu Transformer chính xác hơn thì kết luận gì? | TN |
| Có chạy được trên điện thoại không (kích thước, độ trễ)? | ĐG |
