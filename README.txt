MAMBA CHO BÀN PHÍM GỢI Ý TỪ TIẾNG VIỆT – SO SÁNH VỚI TRANSFORMER
Bài tập lớn môn Nền tảng GenAI (HUST) · Báo cáo buổi 2: 26/12/2026

Huấn luyện language model mức ký tự trên Wikipedia tiếng Việt để gợi ý 3 từ cho bàn phím:
(a) hoàn thành từ đang gõ, (b) gợi ý từ tiếp theo. So sánh Mamba (selective SSM) với Transformer
cùng số tham số (≈ 8M), cùng dữ liệu, về:
  RQ1  độ chính xác gợi ý (BPC, Top-1/Top-3, KSR)
  RQ2  tốc độ huấn luyện và bộ nhớ khi ngữ cảnh dài dần (L = 256 / 1.024 / 4.096)
  RQ3  độ trễ và bộ nhớ khi gợi ý, theo độ dài văn bản đã gõ (256 -> 8.192 ký tự)

Phạm vi: docs/ (Mô tả bài toán thực nghiệm v1, Hướng dẫn Tuần 1 v1), quyết định: docs/decisions.md

NHÓM
  TN  Lê Thị Ngọc Hà (trưởng nhóm)
  DL  Cao Thu Hằng (dữ liệu)
  MB  Nguyễn Tiến Dũng (mô hình Mamba)
  TF  Đặng Duyên Hà (mô hình Transformer)
  ĐG  Ngô Thị Thanh Hằng (huấn luyện & đánh giá)

CẤU TRÚC
  configs/     base.yaml (TN) · mamba.yaml (MB) · transformer.yaml (TF) · smoke.yaml
  data/        prepare.py · loader.py (DL) · synthetic.py (dữ liệu giả cho smoke test)
  models/      __init__.py · base.py (phần dùng chung) · mamba_lm.py (MB) · transformer_lm.py (TF) · dummy.py
  train.py     vòng huấn luyện dùng chung (ĐG)
  bench.py     đo tốc độ – bộ nhớ (ĐG)
  evaluate/    suggest.py · ksr.py · latency.py (ĐG, tuần 2–3)
  demo/        app.py (ĐG, tuần 3)
  tests/       test cho phần của từng người
  notebooks/   00_setup.ipynb (TN) · train_run.ipynb (1 lượt huấn luyện, Save & Run All) · figures.ipynb (tuần 3)
  logs/        CSV log cuối của từng lượt chạy
  docs/        tài liệu nhóm · decisions.md · eval_spec.md · meeting_N7.md · references.md · github_issues_week1.md · slides_outline.md
  utils.py     config, seed, lịch lr, đo thời gian/bộ nhớ, log CSV
  Không đưa lên repo: dữ liệu, checkpoints/, *.pt, *.bin, *.whl

CÀI ĐẶT VÀ CHẠY (máy cá nhân)
  pip install -r requirements.txt
  python -m data.prepare --out data/processed   # tải + xử lý Wikipedia vi (cần Internet)
  bash scripts/smoke_test.sh          # < 1 phút, CPU cũng chạy, in "SMOKE TEST OK"
  python -m pytest -q tests           # giao diện model/dữ liệu
  python train.py --config configs/mamba.yaml --set data.seq_len=1024 --set seed=1
  python train.py --config configs/transformer.yaml --set data.train_file=sample.bin

CHẠY TRÊN KAGGLE
  Import notebooks/00_setup.ipynb vào Kaggle (File -> Import Notebook -> GitHub): GPU T4, Internet On,
  Environment = Pin to original environment (2026-11-19), Add Input 2 Dataset
  mamba-btl-data (DL) và mamba-btl-wheels (MB). Notebook xoá bản clone cũ, clone repo, cài wheel,
  chạy smoke test. Dữ liệu đọc thẳng từ Kaggle Dataset (code tự dò
  /kaggle/input/mamba-btl-data hoặc /kaggle/input/datasets/<owner>/mamba-btl-data).
  Checkpoint và log ghi vào /kaggle/working/{checkpoints,logs}; lượt dài dùng Save Version -> Save & Run All.
  Log CSV cuối của mỗi lượt chép vào logs/ trong repo qua PR.

QUY ƯỚC CHUNG (Hướng dẫn Tuần 1 mục 3)
  Dữ liệu  *.bin uint16 (np.memmap); vocab.json id 0 = <unk>, id 1 = <doc>; NFC, giữ chữ hoa
           get_batch(split, batch_size, seq_len, step, seed) -> x, y int64 (B, L), y = x dịch 1 ký tự
  Model    build_model(cfg.model, vocab_size); model(x) -> logits (B, L, V), loss tính ngoài;
           embedding -> lõi -> RMSNorm -> lm_head (models/base.py); num_params(); step(token, state) ở tuần 2
  Lượt chạy  <model>_L<len>_s<seed>, vd mamba_L1024_s0; batch tính theo ký tự (train.batch_chars)
  Checkpoint checkpoints/<run_name>/ckpt_latest.pt: model, optimizer, scheduler, scaler, step, rng_state, config
  Log CSV    logs/<run_name>.csv: step, train_loss, val_bpc, lr, tokens_per_sec, peak_mem_mb, gpu_name, time

TRẠNG THÁI
  Backend Mamba: cuda (mamba-ssm 2.3.2.post1, wheel ở Dataset mamba-btl-wheels) — chốt 20/11, docs/decisions.md Q7
  Kaggle: GPU T4, Environment = Pin to original environment (2026-11-19) — bắt buộc để wheel khớp torch 2.11/cu130
  Dữ liệu: Dataset mamba-btl-data đã có (vocab 275, train 50M ký tự) — 20/11
  Quy trình Git: CONTRIBUTING.md
