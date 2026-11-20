"""Đo tốc độ – bộ nhớ khi huấn luyện (RQ2). Owner: ĐG.

Chạy (Kaggle, GPU T4):
    python bench.py                                           # 2 model × L ∈ {256, 1024, 4096}, B = batch_chars // L
    python bench.py --seq_lens 4096 --batch_sizes 1 2 4 8     # cố định B để xem bộ nhớ tăng theo B
    python bench.py --seq_lens 8192 16384 --batch_sizes 1     # khám phá: L vượt thiết kế, tìm điểm OOM

Đo một BƯỚC HUẤN LUYỆN đầy đủ như train.py (forward + loss + backward + clip + AdamW, autocast fp16 + GradScaler) trên
token ngẫu nhiên — tốc độ không phụ thuộc nội dung dữ liệu. Thêm thời gian forward không gradient.
- measure(): vài vòng khởi động, torch.cuda.synchronize(), lặp `repeat` lần, lấy TRUNG VỊ (ms). Dùng lại cho
  evaluate/latency.py.
- Bộ nhớ: reset_peak_memory_stats sau khi dựng model, đọc max_memory_allocated sau khi đo (gồm trọng số + optimizer
  + activation). OOM -> ghi "OOM" rồi đo cấu hình tiếp theo, không dừng chương trình.
- Luôn ghi tên GPU, phiên bản torch. Mamba đo bằng backend "cuda" (mamba-ssm) — backend "torch" chỉ để chạy thử.
Kết quả: logs/bench_<gpu>_<YYYYmmdd-HHMM>.csv (trên Kaggle tự đặt dưới /kaggle/working/logs).
"""
from __future__ import annotations

import argparse
import csv
import gc
import statistics
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from models import build_model
from train import out_dir, param_groups
from utils import Timer, get_device, gpu_name, load_config, peak_mem_mb, sync, to_ns

FIELDS = ["model", "backend", "seq_len", "batch_size", "chars_per_step", "params", "precision", "train_step_ms",
          "fwd_ms", "chars_per_sec", "peak_mem_mb", "status", "gpu_name", "torch", "time"]


def measure(fn, device: torch.device, warmup: int = 3, repeat: int = 10) -> float:
    """Trung vị thời gian chạy fn() (ms), có khởi động và cuda.synchronize."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeat):
        with Timer(device) as t:
            fn()
        times.append(t.elapsed * 1000)
    return statistics.median(times)


def bench_one(cfg, seq_len: int, batch_size: int, vocab_size: int, device, args) -> dict:
    row = {"model": cfg.model.name, "backend": getattr(cfg.model, "backend", "-"), "seq_len": seq_len,
           "batch_size": batch_size, "chars_per_step": seq_len * batch_size, "precision": cfg.train.precision}
    model = opt = None
    try:
        torch.manual_seed(0)
        model = build_model(cfg.model, vocab_size).to(device)
        row["params"] = model.num_params()
        use_fp16 = cfg.train.precision == "fp16" and device.type == "cuda"
        amp = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else nullcontext
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        opt = torch.optim.AdamW(param_groups(model, cfg.train.weight_decay), lr=1e-4)
        x = torch.randint(vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(vocab_size, (batch_size, seq_len), device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        def train_step():
            opt.zero_grad(set_to_none=True)
            with amp():
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, vocab_size), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

        @torch.no_grad()
        def fwd():
            with amp():
                model(x)

        model.train()
        row["train_step_ms"] = round(measure(train_step, device, args.warmup, args.repeat), 2)
        row["peak_mem_mb"] = round(peak_mem_mb(device), 1) if device.type == "cuda" else ""
        model.eval()
        row["fwd_ms"] = round(measure(fwd, device, args.warmup, args.repeat), 2)
        row["chars_per_sec"] = round(seq_len * batch_size / (row["train_step_ms"] / 1000))
        row["status"] = "ok"
    except torch.cuda.OutOfMemoryError:
        row["status"] = "OOM"
    except RuntimeError as e:  # một số kernel báo hết bộ nhớ bằng RuntimeError
        if "out of memory" not in str(e).lower():
            raise
        row["status"] = "OOM"
    finally:
        del model, opt
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            sync(device)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["configs/transformer.yaml", "configs/mamba.yaml"])
    ap.add_argument("--seq_lens", nargs="+", type=int, default=[256, 1024, 4096])
    ap.add_argument("--batch_sizes", nargs="+", type=int, default=None,
                    help="cố định B; mặc định B = train.batch_chars // L như lúc huấn luyện")
    ap.add_argument("--vocab_size", type=int, default=275, help="= meta.json của mamba-btl-data")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default=None, help="ghi đè train.precision")
    ap.add_argument("--backend", choices=["cuda", "torch"], default=None, help="ghi đè backend Mamba")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--out", default=None, help="mặc định logs/bench_<gpu>_<thời gian>.csv")
    args = ap.parse_args()

    device = get_device()
    gname = gpu_name(device)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = Path(args.out) if args.out else out_dir("logs") / f"bench_{gname.replace(' ', '')}_{stamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={gname} torch={torch.__version__} -> {out}")

    rows = []
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for path in args.configs:
            cfg = to_ns(load_config(path))
            if args.precision:
                cfg.train.precision = args.precision
            if args.backend and cfg.model.name == "mamba":
                cfg.model.backend = args.backend
            for L in args.seq_lens:
                for B in args.batch_sizes or [max(1, cfg.train.batch_chars // L)]:
                    t0 = time.perf_counter()
                    row = bench_one(cfg, L, B, args.vocab_size, device, args)
                    row.update(gpu_name=gname, torch=torch.__version__, time=datetime.now().isoformat(timespec="seconds"))
                    w.writerow({k: row.get(k, "") for k in FIELDS})
                    f.flush()
                    rows.append(row)
                    print(f"{row['model']:<12} L={L:<6} B={B:<4} {row['status']:<4} "
                          f"step={row.get('train_step_ms', '-'):>9} ms  fwd={row.get('fwd_ms', '-'):>8} ms  "
                          f"{row.get('chars_per_sec', '-'):>9} char/s  mem={row.get('peak_mem_mb', '-'):>8} MB  "
                          f"({time.perf_counter() - t0:.0f}s)")
    print(f"Đã ghi {len(rows)} dòng -> {out}")


if __name__ == "__main__":
    main()
