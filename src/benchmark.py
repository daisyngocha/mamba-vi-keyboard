"""Benchmark hiệu năng theo độ dài chuỗi (thí nghiệm E3). Owner: Cao Thanh Hằng.

Chạy:  python -m src.benchmark --configs configs/transformer.yaml configs/mamba.yaml \
                                --seq_lens 256 512 1024 2048 4096 --batch_size 8

Với mỗi (model, L):
- Khởi tạo model ngẫu nhiên (không cần train), input ngẫu nhiên [B, L].
- Warmup 3 lần, đo 10 lần: thời gian forward, forward+backward (dùng src.utils.Timer — có cuda.synchronize).
- Peak memory: torch.cuda.reset_peak_memory_stats() trước mỗi cấu hình, đọc src.utils.peak_mem_mb() sau.
- Bắt torch.cuda.OutOfMemoryError -> ghi "OOM" (đây cũng là kết quả cần báo cáo).
- Latency suy luận: batch 1, sinh 1 ký tự với context L.
Ghi ra results/benchmark/<gpu>_<date>.csv: model, impl, seq_len, batch, fwd_ms, fwd_bwd_ms, peak_mem_mb, status.

Kèm generate_selective_copying(n, seq_len=4096, n_tokens=16, vocab=16) cho thí nghiệm E1
(theo setup Table 1 của paper: 16 token nội dung rải ngẫu nhiên giữa token nhiễu, model phải chép lại đúng thứ tự).
"""
raise SystemExit("TODO(Thanh Hằng): cài đặt benchmark.py")
