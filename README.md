# Mamba vs Transformer — Gợi ý từ bàn phím tiếng Việt

BTL môn **Nền tảng GenAI** (HUST). Cài đặt Mamba (selective SSM), so sánh với Transformer trên language model mức ký tự tiếng Việt về độ chính xác, tốc độ huấn luyện và bộ nhớ.

Nhóm: Lê Thị Ngọc Hà (trưởng nhóm)
      Đặng Duyên Hà
      Nguyễn Tiến Dũng
      Ngô Thị Thu Hằng
      Cao Thanh Hằng
Phạm vi chi tiết: [`docs/scope.md`](docs/scope.md).

## Cấu trúc
```
configs/          base.yaml (chung) + transformer.yaml, mamba.yaml, smoke.yaml
data/             prepare_data.py (tải + làm sạch + encode); data/processed/ không commit
src/models/       dummy.py, transformer.py, mamba_pt.py, mamba_ssm_wrap.py (+ registry)
src/dataset.py    CharVocab, đọc .bin, get_batch
src/train.py      vòng train chung -> results/<run>/
src/evaluate.py   BPC, Top-k word accuracy, KSR
src/benchmark.py  thời gian/bộ nhớ theo độ dài chuỗi, Selective Copying
docs/             scope, agenda, dàn ý slide
```

## Chạy nhanh
```bash
pip install -r requirements.txt
bash scripts/smoke_test.sh                               # < 1 phút, CPU cũng chạy
python -m src.train --config configs/mamba.yaml          # sau khi có data + model
python -m src.train --config configs/transformer.yaml --set train.lr=3e-4 --set seed=1
```

## Chạy trên Kaggle
1. New Notebook → Settings: Accelerator **GPU T4 x2** (hoặc P100), Internet **On** (cần xác minh số điện thoại).
2. Cell đầu:
   ```bash
   %cd /kaggle/working
   !rm -rf mamba-vi-keyboard
   !git clone -q https://github.com/daisyngocha/mamba-vi-keyboard.git
   %cd /kaggle/working/mamba-vi-keyboard
   !pip install -q datasets pyyaml
   !bash scripts/smoke_test.sh
   ```
3. Kết quả nằm ở `results/<run>/`; tải `summary.json` + `metrics.csv` về, **không** commit checkpoint.
   # summary.json (kết quả chạy smoke test trên kagle)
   {
  "run_name": "smoke",
  "model": "dummy",
  "params": 10739,
  "steps": 60,
  "best_val_loss": 2.9529,
  "best_val_bpc": 4.2602,
  "mean_tokens_per_s": 317923,
  "peak_mem_mb": 18.5,
  "train_time_s": 0.5,
  "device": "Tesla T4"
   }
   # metrics.csv (kết quả chạy smoke test trên kagle)
   step	split	loss	bpc	lr	tokens_per_s	peak_mem_mb	elapsed_s
   0	10	train	3.7895	5.4671	0.001000	32816.0	18.0	0.3
   1	20	train	3.5495	5.1208	0.000930	374550.0	18.0	0.3
   2	30	train	3.2821	4.7351	0.000716	396410.0	18.0	0.4
   3	30	val	3.2783	4.7295	NaN	NaN	NaN	0.4
   4	40	train	3.0901	4.4581	0.000438	325266.0	18.0	0.4
   5	50	train	2.9982	4.3255	0.000203	387613.0	18.0	0.4
   6	60	train	2.9647	4.2771	0.000101	390883.0	18.0	0.4
   7	60	val	2.9529	4.2602	NaN	NaN	NaN	0.5

## Interface chung (bắt buộc)
- Model: `build_model(cfg.model, vocab_size)`; `model(idx[B,T]) -> logits[B,T,V]`, causal, không tự tính loss.
- Dữ liệu: `data/processed/{vocab.json, train.bin, val.bin, test.bin}` (uint16).
- Kết quả: mỗi run có `config.yaml`, `metrics.csv`, `summary.json` — bảng trên slide sinh từ đây.

Quy trình làm việc: [`CONTRIBUTING.md`](CONTRIBUTING.md).
