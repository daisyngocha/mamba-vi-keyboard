"""Công cụ đo tốc độ – bộ nhớ dùng chung. Owner: ĐG (Cao Thanh Hằng).

Chạy:  python bench.py --configs configs/mamba.yaml configs/transformer.yaml --seq_lens 256 1024 4096

Đặc tả — Hướng dẫn Tuần 1 mục 8 (RQ2: chi phí huấn luyện theo L):
- Hàm đo dùng chung measure(fn, device, warmup=3, repeat=10) -> trung vị (ms):
  vài vòng khởi động, torch.cuda.synchronize() (utils.Timer), lặp nhiều lần lấy trung vị.
- Bộ nhớ: torch.cuda.reset_peak_memory_stats() trước mỗi cấu hình, đọc utils.peak_mem_mb() sau.
- Bắt torch.cuda.OutOfMemoryError -> ghi "OOM" thay vì dừng chương trình (OOM cũng là kết quả cần báo cáo).
- Với mỗi (model, L): batch_size = batch_chars // L như lúc train; đo forward+backward và ký tự/giây.
- Luôn ghi tên GPU. Ghi CSV vào logs/bench_<gpu>_<ngày>.csv:
      model, backend, seq_len, batch_size, fwd_bwd_ms, chars_per_sec, peak_mem_mb, gpu_name, status
- Đo độ trễ gợi ý theo độ dài văn bản đã gõ (RQ3) nằm ở evaluate/latency.py (tuần 2–3), dùng lại measure().
"""
raise SystemExit("TODO(ĐG): cài đặt bench.py")
