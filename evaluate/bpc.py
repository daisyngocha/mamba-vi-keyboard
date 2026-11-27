"""BPC trên toàn bộ test.bin (RQ1) — đúng docs/eval_spec.md mục 1. Issue #24.

- Cắt test.bin thành các cửa sổ KHÔNG chồng nhau dài L: cửa sổ i có đầu vào ids[iL : (i+1)L], nhãn
  ids[iL+1 : (i+1)L+1] -> mỗi ký tự được chấm đúng 1 lần (trừ ký tự đầu tiên của file), bỏ phần dư cuối.
- BPC = Σ −ln p(x_t | x_<t) / (N · ln 2), N = số ký tự được chấm (kể cả <doc>, <unk>).
- Ký tự đầu mỗi cửa sổ không có ngữ cảnh -> BPC test cao hơn val BPC lúc train một chút; đây là quy ước chung
  cho mọi model (cùng L thì cùng cửa sổ), nên so sánh Mamba vs Transformer vẫn công bằng.
- Lưu BPC từng cửa sổ (`per_window`) để kiểm định ghép cặp Mamba vs Transformer cùng L: các cửa sổ cùng độ dài nên
  BPC tổng = trung bình BPC các cửa sổ -> kiểm định t ghép cặp trên cửa sổ kiểm định đúng chênh lệch BPC tổng.

Gọi qua `python -m evaluate.ksr --run <run>` (chấm luôn cả Top-k, KSR) hoặc riêng:
    python -m evaluate.bpc --run mamba_L1024_s0
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from evaluate.common import autocast_ctx

TOKENS_PER_BATCH = 16_384       # số ký tự mỗi batch: L256 -> 64 cửa sổ, L4096 -> 4 cửa sổ (vừa T4 cả 2 model)


def n_windows(n_ids: int, seq_len: int) -> int:
    """Số cửa sổ không chồng dài seq_len (cần thêm 1 id làm nhãn cho vị trí cuối)."""
    return max(0, (n_ids - 1) // seq_len)


@torch.no_grad()
def bpc_full(model, ids, seq_len: int, dtype: str = "fp16", batch_size: int | None = None,
             log_every: int = 0) -> dict:
    """BPC của model trên toàn bộ dãy ids (np.memmap/np.ndarray uint16), cửa sổ không chồng dài seq_len."""
    device = next(model.parameters()).device
    n_win = n_windows(len(ids), seq_len)
    if n_win == 0:
        raise ValueError(f"Dữ liệu ({len(ids)} ký tự) ngắn hơn một cửa sổ L = {seq_len}")
    bs = batch_size or max(1, TOKENS_PER_BATCH // seq_len)
    arr = np.asarray(ids[: n_win * seq_len + 1], dtype=np.int64)
    nll_win = np.empty(n_win, dtype=np.float64)          # Σ −ln p trên từng cửa sổ (nats)
    t0 = time.perf_counter()
    for b0 in range(0, n_win, bs):
        b1 = min(n_win, b0 + bs)
        x = torch.from_numpy(np.stack([arr[i * seq_len: (i + 1) * seq_len] for i in range(b0, b1)])).to(device)
        y = torch.from_numpy(np.stack([arr[i * seq_len + 1: (i + 1) * seq_len + 1] for i in range(b0, b1)])).to(device)
        with autocast_ctx(device, dtype):
            logits = model(x)
        nll = F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none")      # (B, L), fp32
        nll_win[b0:b1] = nll.double().sum(1).cpu().numpy()
        if log_every and (b1 // bs) % log_every == 0:
            print(f"  BPC {b1}/{n_win} cửa sổ | tạm tính {nll_win[:b1].sum() / (b1 * seq_len) / math.log(2):.4f} "
                  f"| {time.perf_counter() - t0:.0f} s", flush=True)
    n = n_win * seq_len
    return {"bpc": float(nll_win.sum() / n / math.log(2)), "chars": int(n), "windows": int(n_win),
            "seq_len": int(seq_len), "dropped_tail": int(len(ids) - 1 - n), "batch_size": int(bs),
            "time_s": round(time.perf_counter() - t0, 1),
            "per_window": [round(float(v), 5) for v in nll_win / seq_len / math.log(2)]}


def main():
    from pathlib import Path

    from data.loader import CharVocab, resolve_data_dir
    from evaluate.common import find_checkpoint, load_run, write_eval
    from train import out_dir

    ap = argparse.ArgumentParser(description="BPC trên toàn bộ test.bin cho một checkpoint (eval_spec mục 1)")
    ap.add_argument("--run")
    ap.add_argument("--ckpt")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--l_ctx", type=int, default=None, help="độ dài cửa sổ; mặc định = L lúc huấn luyện")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--log_dir", default="logs")
    args = ap.parse_args()
    ckpt = Path(args.ckpt) if args.ckpt else find_checkpoint(args.run, args.checkpoint_dir)
    if ckpt is None or not ckpt.exists():
        raise SystemExit(f"Không thấy checkpoint của {args.run!r}")
    model, vocab, info = load_run(ckpt)
    data_dir = resolve_data_dir(args.data_dir)
    assert CharVocab.load(data_dir / "vocab.json").itos == vocab.itos, "vocab checkpoint khác vocab.json của dữ liệu"
    run_bpc(model, args.run or info["run_name"], info, data_dir, args.l_ctx or info["seq_len"], args.dtype,
            out_dir(args.log_dir), write_eval)


def run_bpc(model, run_name, info, data_dir, l_ctx, dtype, log_dir, write_eval, common: dict | None = None) -> dict:
    """Chấm BPC test và ghi khóa "bpc" (hoặc "bpc_ctx<l_ctx>" khi đổi ngữ cảnh) vào logs/<run>_eval.json."""
    device = next(model.parameters()).device
    test_ids = np.memmap(data_dir / "test.bin", dtype=np.uint16, mode="r")
    res = bpc_full(model, test_ids, l_ctx, dtype, log_every=20)
    suffix = "" if l_ctx == info["seq_len"] else f"_ctx{l_ctx}"
    common = common or {"checkpoint_step": info["step"], "l_ctx": l_ctx, "dtype": dtype,
                        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}
    write_eval(run_name, "bpc" + suffix, {**common, **res}, log_dir)
    print(f"BPC test {res['bpc']:.4f} ({res['chars']:,} ký tự, {res['windows']} cửa sổ × {l_ctx}, "
          f"{res['time_s']} s)", flush=True)
    return res


if __name__ == "__main__":
    main()
