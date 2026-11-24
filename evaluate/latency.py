"""Độ trễ và bộ nhớ khi gợi ý (RQ3). Owner: ĐG (Ngô Thị Thanh Hằng). Tuần 3.

Đặc tả — Mô tả bài toán mục 4.1–4.3:
- Độ dài văn bản đã gõ: 256 -> 8.192 ký tự. Với mỗi độ dài: nạp văn bản bằng model.step(...) (Mamba: trạng thái
  cố định; Transformer: KV cache), rồi đo ms để ra 3 gợi ý sau 1 lần gõ phím (dùng bench.measure, lấy trung vị).
- Bộ nhớ khi chạy: peak memory theo độ dài văn bản đã gõ — kỳ vọng Mamba không đổi, Transformer tăng dần.
- Kích thước model (MB). Tùy chọn: đo trên CPU (sinh từng bước) để gần điều kiện điện thoại.
- Ghi CSV vào logs/latency_<gpu>_<ngày>.csv.
"""
raise SystemExit("TODO(ĐG): cài đặt evaluate/latency.py (tuần 3)")
