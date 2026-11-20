# Kết quả bench.py — Tesla T4, 20/11/2026 (tuần 1)

torch 2.11.0+cu130 · fp16 · Mamba backend `cuda` (mamba-ssm 2.3.2.post1) · Transformer SDPA memory-efficient · ≈ 7,2M tham số mỗi model.
Bước huấn luyện đầy đủ (forward + backward + AdamW), token ngẫu nhiên, trung vị 10 lần sau 3 lần khởi động.

**Theo L, cùng 32.768 ký tự/bước** (`python bench.py`, 3 lần chạy: 0737, 0739, 0740 — bảng lấy trung vị)

| L | B | Transformer ký tự/s | Mamba ký tự/s | Mamba / Transformer | Bộ nhớ đỉnh T / M (MB) | Forward T / M (ms) |
|---|---|---|---|---|---|---|
| 256 | 128 | 138.489 | 65.623 | 0,47 | 2.728 / 2.742 | 77 / 106 |
| 1.024 | 32 | 119.801 | 73.838 | 0,62 | 2.729 / 2.669 | 85 / 98 |
| 4.096 | 8 | 80.813 | 79.934 | 0,99 | 2.732 / 2.662 | 115 / 96 |

Tỷ lệ Mamba / Transformer ổn định giữa 3 lần chạy (0,47–0,50 · 0,61–0,67 · 0,92–0,99). Lần chạy đầu (0737) nhanh hơn ~15% ở cả hai model (GPU còn nguội) → so sánh trong cùng một lần chạy, báo trung vị nhiều lần.

**Khám phá: L vượt thiết kế, vẫn 32.768 ký tự/bước** (`--seq_lens 8192 16384`, 1 lần chạy 0749 — cần thêm 2 lần để lấy trung vị)

| L | B | Transformer ký tự/s | Mamba ký tự/s | Mamba / Transformer | Bộ nhớ đỉnh T / M (MB) | Forward T / M (ms) |
|---|---|---|---|---|---|---|
| 8.192 | 4 | 55.735 | 84.724 | 1,52 | 2.736 / 2.662 | 160 / 92 |
| 16.384 | 2 | 33.714 | 85.516 | 2,54 | 2.746 / 2.662 | 225 / 91 |

**Theo B, L = 4.096** (`--seq_lens 4096 --batch_sizes 1 2 4 8`, 2 lần chạy: 0738, 0741)

| B | Transformer ký tự/s · MB | Mamba ký tự/s · MB |
|---|---|---|
| 1 | 78.960 · 453 | 77.940 · 439 |
| 2 | 83.981 · 786 | 83.929 · 778 |
| 4 | 86.439 · 1.424 | 83.383 · 1.424 |
| 8 | 87.243 · 2.732 | 82.254 · 2.661 |

**Nhận xét sơ bộ (RQ2)**
- Khi L tăng 256 → 16.384: Transformer chậm dần (138k → 120k → 81k → 56k → 34k ký tự/s, tức ≈ tỉ lệ nghịch với L ở L lớn — chi phí attention O(L²) trên mỗi chuỗi), Mamba tăng rồi đi ngang ≈ 80–85k từ L = 1.024 (chi phí O(L)). Điểm giao ≈ L = 4.096; ở L = 16.384 Mamba nhanh gấp 2,5 lần.
- Ở L ≤ 1.024 Transformer nhanh hơn (gấp 2 ở L = 256): trên T4, attention ngắn rẻ hơn selective scan.
- Bộ nhớ huấn luyện gần như không đổi theo L và bằng nhau giữa 2 model (≈ 2,7 GB): attention memory-efficient chỉ tốn bộ nhớ tuyến tính theo L, và số ký tự mỗi bước cố định. Kể cả ở L = 16.384 bộ nhớ vẫn ≈ 2,7 GB cho cả 2 model: khác biệt O(L²) của attention không thể hiện ở bộ nhớ huấn luyện khi dùng kernel này, chỉ thể hiện ở thời gian.
- Bộ nhớ tăng tuyến tính theo B (~330 MB mỗi chuỗi 4.096 ký tự), không OOM; còn xa giới hạn 15 GB.
