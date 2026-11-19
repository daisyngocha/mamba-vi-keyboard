# GitHub Issues — Tuần 1 (20–26/11)

Mẫu nội dung các issue tuần 1. Số issue khớp với GitHub (#1 là PR khởi tạo repo).
Label: `dataset` / `model` / `evaluation` / `leader` · Milestone: `Deadline Tuần 1 (Chuẩn bị môi trường + Smoke test)`.
Deadline PR chung: **CN 23/11 22:00**. Sync ghép pipeline: **T2 24/11 21:00**. Tổng kết: **T4 26/11 21:00**.

---

### #2 [Data] Chuẩn bị dữ liệu Wikipedia tiếng Việt
**Assignee:** Ngô Thị Thu Hằng · **Deadline PR:** CN 23/11
Cài đặt `data/prepare_data.py` theo docstring trong file và `docs/scope.md` mục 3.
**Hoàn thành khi:**
- [ ] `data/processed/` có `vocab.json`, `train/val/test.bin`, `stats.json`, `wordsuggest_test.jsonl` (2.000 dòng)
- [ ] Chia theo bài viết, seed 42; vocab 100–150 ký tự; tỉ lệ `<unk>` < 0.1%
- [ ] Đã NFC-normalize; in 5 mẫu decode để kiểm tra bằng mắt (dán vào PR)
- [ ] `python -m src.train --config configs/mamba.yaml --set model.name=dummy --set train.max_steps=100` chạy được với data thật
- [ ] Upload `data/processed/` lên **Kaggle Dataset** (private, share cho nhóm) để mọi người khỏi xử lý lại

### #3 [Model] Transformer baseline + hoàn thiện train.py
**Assignee:** Đặng Duyên Hà · **Deadline PR:** CN 23/11
Cài đặt `src/models/transformer.py` theo docstring.
**Hoàn thành khi:**
- [ ] Đúng interface `forward(idx[B,T]) -> logits[B,T,V]`, causal (có unit test đổi token tương lai không đổi logits quá khứ)
- [ ] In số tham số với `configs/transformer.yaml` (mục tiêu ~3–3.5M)
- [ ] Overfit 1 batch (loss < 0.1) — dán log
- [ ] Chạy được với `max_seq_len` 4096 (cho benchmark)
- [ ] `mean_tokens_per_s` trong `summary.json` bỏ khoảng log đầu tiên (CUDA warmup)
- [ ] Đề xuất cải tiến `train.py` nếu cần (vd gradient accumulation) — mở PR riêng

### #4 [Model] Mamba tự cài đặt (PyTorch thuần) + thử mamba-ssm
**Assignee:** Nguyễn Tiến Dũng · **Deadline PR:** CN 23/11 (mamba_pt), T4 26/11 (mamba_ssm)
Cài đặt `src/models/mamba_pt.py` theo docstring (Section 3 + Figure 3 paper).
**Hoàn thành khi:**
- [ ] Đúng interface, causal (unit test)
- [ ] Số tham số với `configs/mamba.yaml` lệch < 10% so với Transformer; báo lại `n_layer` đã chỉnh
- [ ] Overfit 1 batch (loss < 0.1) — dán log
- [ ] Ghi chú: thời gian 1 bước train seq_len 256 trên T4
- [ ] Thử `pip install mamba-ssm causal-conv1d` trên Kaggle (tối đa 2 giờ); thành công → `mamba_ssm_wrap.py` + so khớp output; thất bại → dán log lỗi vào issue

### #5 [Eval] Bộ đánh giá + benchmark + Selective Copying
**Assignee:** Cao Thanh Hằng · **Deadline PR:** CN 23/11 (evaluate), T4 26/11 (benchmark)
Cài đặt `src/evaluate.py` và `src/benchmark.py` theo docstring. Dùng checkpoint smoke test để phát triển trước khi có model thật.
**Hoàn thành khi:**
- [ ] `evaluate.py` ra BPC test + Top-1/Top-3@k0,1,2 + KSR trên checkpoint bất kỳ
- [ ] `benchmark.py` chạy với model `dummy`, xuất CSV đúng cột; xử lý OOM
- [ ] `generate_selective_copying()` + 3 mẫu in ra kiểm tra bằng mắt
- [ ] Hàm vẽ biểu đồ thời gian/bộ nhớ theo seq_len (matplotlib, log-scale trục x)
- [ ] `notebooks/make_figures.ipynb` đọc `results/*/summary.json` + `metrics.csv`, vẽ biểu đồ cho slide; chạy được với kết quả smoke test

### #6 [Lead] Điều phối + dàn ý slide
**Assignee:** Lê Thị Ngọc Hà
- [ ] Chốt scope sau kickoff (scope.md v1.0)
- [ ] Review mọi PR trong 24h
- [ ] `docs/slides_outline.md` v1.0, phân người viết từng phần
- [ ] T2 24/11: ghép pipeline, chạy 200 bước mỗi model
- [ ] T3–T4: sanity test + train ngắn (~1h GPU/model) + benchmark lần đầu; tổng kết tuần 1
