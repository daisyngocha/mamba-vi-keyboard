"""Độ trễ và bộ nhớ khi gợi ý theo độ dài văn bản đã gõ (RQ3). Owner: MB (Tiến Dũng). docs/eval_spec.md mục 5.

Với mỗi T ∈ {256, 512, 1.024, 2.048, 4.096, 8.192}:
- Ngữ cảnh = T ký tự của test.bin đứng ngay TRƯỚC vị trí S cố định; nạp trước bằng prefill() (cho cùng state với
  lặp step(), tests/test_*_step.py) — thời gian nạp không tính.
- Gõ tiếp các ký tự test.bin[S : S + 5 + 50]: mỗi phím = step(ký tự mới) + suggest (beam 5, ≤ 10 ký tự, 3 từ).
  5 phím khởi động, đo 50 phím liên tiếp, batch 1, cuda.synchronize hai đầu -> trung vị (và p90).
  Mọi T gõ CÙNG 55 phím đó (S cố định) -> chỉ độ dài ngữ cảnh thay đổi.
- Không cắt ngữ cảnh: Transformer giữ KV cache dài T kể cả T > L lúc huấn luyện (eval_spec mục 5: độ trễ, bộ nhớ
  đo với cache dài T; chất lượng gợi ý khi T > L không dùng ở đây).
- Bộ nhớ: peak_mem_mb = torch.cuda.max_memory_allocated trong 50 phím đo (gồm trọng số, state, beam);
  state_mb = state_nbytes của state batch 1 ngay sau khi nạp T ký tự (Mamba cố định, Transformer ∝ T);
  model_mb = số tham số × 2 byte (fp16).
- Tách step_ms (cập nhật state khi gõ 1 phím) và suggest_ms (beam search). Beam search nhân bản state thành 5 nhánh:
  với Transformer là chép KV cache dài T × 5 mỗi bước beam — chi phí thật của cách cài đặt hiện tại.

Ghi logs/latency/<run>_<thiết bị>.csv (mỗi dòng 1 T). Chạy (thư mục gốc repo; Kaggle: Add Input checkpoint):
    python -m evaluate.latency --run mamba_L1024_s0                   # GPU, fp16 autocast
    python -m evaluate.latency --run transformer_L1024_s0 --T 256 1024 4096
    python -m evaluate.latency --run mamba_L1024_s0 --device cpu --threads 1     # CPU fp32 1 luồng (gần điện thoại)
"""
from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from evaluate.common import autocast_ctx, is_letter
from evaluate.suggest import suggest_from_state
from utils import Timer

TS = [256, 512, 1024, 2048, 4096, 8192]
START = 20_000                  # vị trí S trong test.bin (cố định, > max T) — 55 phím gõ bắt đầu từ đây
WARMUP, N_KEYS = 5, 50
SUGGEST_CFG = {"k": 3, "beam": 5, "max_word_len": 10}
FIELDS = ["run", "model", "train_L", "T", "device", "dtype", "step_ms", "suggest_ms", "total_ms", "total_p90_ms",
          "peak_mem_mb", "state_mb", "state_formula_mb", "model_mb", "params", "n_keys", "prefill_s", "time"]
MB = 2 ** 20


def state_formula_bytes(model, info: dict, T: int, bytes_per: int = 2) -> int:
    """Công thức eval_spec mục 5 (fp16). Mamba dùng d_conv (conv_state giữ d_conv cột; xem models/mamba_lm.py)."""
    cfg = model.cfg
    if info["model"] == "transformer":
        return 2 * cfg.n_layer * T * cfg.d_model * bytes_per
    d_inner = getattr(cfg, "expand", 2) * cfg.d_model
    return cfg.n_layer * d_inner * (getattr(cfg, "d_state", 16) + getattr(cfg, "d_conv", 4)) * bytes_per


def letters_tail(chars: list[str]) -> str:
    """Prefix của từ đang gõ = các chữ cái liên tiếp ở cuối (token <doc>/<unk> là ký tự kết thúc từ)."""
    out = []
    for c in reversed(chars):
        if not is_letter(c):
            break
        out.append(c)
    return "".join(reversed(out))


@torch.no_grad()
def measure_T(model, vocab, ids: np.ndarray, T: int, dtype: str, start: int = START,
              warmup: int = WARMUP, n_keys: int = N_KEYS) -> dict:
    """Độ trễ 1 phím (step + suggest) sau khi đã nạp T ký tự, trung vị n_keys phím liên tiếp."""
    device = next(model.parameters()).device
    assert start >= T and start + warmup + n_keys <= len(ids), "test.bin không đủ dài cho T/start này"
    ctx = torch.as_tensor(np.asarray(ids[start - T:start], dtype=np.int64), device=device)[None]
    keys = [int(i) for i in ids[start:start + warmup + n_keys]]
    chars = [vocab.itos[i] for i in ids[start - 32:start]]          # đủ để tách prefix của từ đang gõ
    with autocast_ctx(device, dtype):
        t0 = time.perf_counter()
        logits, state = model.prefill(ctx)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prefill_s = time.perf_counter() - t0
        state_b = model.state_nbytes(state)
        step_ms, sugg_ms = [], []
        for j, tok in enumerate(keys):
            if j == warmup and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            with Timer(device) as t1:
                logits, state = model.step(torch.tensor([tok], device=device), state)
            chars.append(vocab.itos[tok])
            with Timer(device) as t2:
                suggest_from_state(model, vocab, logits, state, letters_tail(chars), **SUGGEST_CFG)
            if j >= warmup:
                step_ms.append(t1.elapsed * 1000)
                sugg_ms.append(t2.elapsed * 1000)
    total = [a + b for a, b in zip(step_ms, sugg_ms)]
    return {"T": T, "step_ms": statistics.median(step_ms), "suggest_ms": statistics.median(sugg_ms),
            "total_ms": statistics.median(total), "total_p90_ms": float(np.percentile(total, 90)),
            "peak_mem_mb": torch.cuda.max_memory_allocated(device) / MB if device.type == "cuda" else float("nan"),
            "state_mb": state_b / MB, "n_keys": n_keys, "prefill_s": round(prefill_s, 3)}


def main():
    from data.loader import CharVocab, resolve_data_dir
    from evaluate.common import find_checkpoint, load_run
    from train import out_dir

    ap = argparse.ArgumentParser(description="Độ trễ / bộ nhớ khi gợi ý theo độ dài văn bản đã gõ (RQ3)")
    ap.add_argument("--run")
    ap.add_argument("--ckpt")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--T", type=int, nargs="+", default=TS)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default=None, help="mặc định: GPU fp16, CPU fp32")
    ap.add_argument("--threads", type=int, default=None, help="số luồng CPU (eval_spec: 1 luồng)")
    ap.add_argument("--log_dir", default="logs")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = args.dtype or ("fp16" if device.type == "cuda" else "fp32")
    if args.threads:
        torch.set_num_threads(args.threads)
    ckpt = Path(args.ckpt) if args.ckpt else find_checkpoint(args.run, args.checkpoint_dir)
    if ckpt is None or not ckpt.exists():
        raise SystemExit(f"Không thấy checkpoint của {args.run!r}. Trên Kaggle: Add Input output notebook của lượt đó.")
    model, vocab, info = load_run(ckpt, device)
    run = args.run or info["run_name"]
    data_dir = resolve_data_dir(args.data_dir)
    assert CharVocab.load(data_dir / "vocab.json").itos == vocab.itos, "vocab checkpoint khác vocab.json của dữ liệu"
    ids = np.memmap(data_dir / "test.bin", dtype=np.uint16, mode="r")

    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else f"cpu{args.threads or ''}t"
    tag = "".join(c for c in dev_name.lower().replace("tesla ", "") if c.isalnum())
    out = out_dir(args.log_dir) / "latency" / f"{run}_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    params = model.num_params()
    print(f"[{run}] {info['model']} L={info['seq_len']} backend {info['backend']} | {dev_name} {dtype} | "
          f"{params / 1e6:.2f}M tham số -> {out}", flush=True)
    rows = []
    for T in sorted(args.T):
        try:
            r = measure_T(model, vocab, ids, T, dtype)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  T={T:>5}: OOM", flush=True)
            continue
        r.update({"run": run, "model": info["model"], "train_L": info["seq_len"], "device": dev_name, "dtype": dtype,
                  "state_formula_mb": state_formula_bytes(model, info, T) / MB, "model_mb": params * 2 / MB,
                  "params": params, "time": time.strftime("%Y-%m-%d %H:%M")})
        rows.append(r)
        print(f"  T={T:>5}: {r['total_ms']:6.1f} ms/phím (step {r['step_ms']:5.2f} + suggest {r['suggest_ms']:6.1f}, "
              f"p90 {r['total_p90_ms']:6.1f}) | peak {r['peak_mem_mb']:7.1f} MB | state {r['state_mb']:7.2f} MB | "
              f"nạp {r['prefill_s']} s", flush=True)
        with open(out, "w", newline="", encoding="utf-8") as f:          # ghi lại sau mỗi T (chạy dở vẫn còn số)
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()} for row in rows)


if __name__ == "__main__":
    main()
