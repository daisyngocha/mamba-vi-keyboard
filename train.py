"""Vòng huấn luyện dùng chung cho mọi model. Owner: ĐG (Cao Thanh Hằng); khung ban đầu do TN viết.

Chạy:
    python train.py --config configs/smoke.yaml
    python train.py --config configs/mamba.yaml --set data.seq_len=1024 --set seed=1

Đầu ra (mục 3.4):
    <checkpoint_dir>/<run_name>/ckpt_latest.pt     (không commit; trên Kaggle tự đặt dưới /kaggle/working)
    <log_dir>/<run_name>.csv                       cột: step, train_loss, val_bpc, lr, tokens_per_sec,
                                                   peak_mem_mb, gpu_name, time
    <log_dir>/<run_name>_summary.json

Đã có: đọc config -> tạo model -> vòng lặp get_batch/forward/CE loss/backward/AdamW, lr cosine có warmup,
fp16 autocast + GradScaler, clip gradient, batch tính theo ký tự, đánh giá val trên tập batch cố định, BPC, log CSV.

TODO(ĐG) — Hướng dẫn Tuần 1 mục 8:
  [ ] save_checkpoint: dict đủ khóa model, optimizer, scheduler, scaler, step, rng_state, config;
      ghi ra file tạm rồi os.replace để không hỏng file khi Kaggle ngắt giữa lúc ghi.
  [ ] find_resume_checkpoint: tìm ckpt_latest.pt của run_name trong checkpoint_dir VÀ dưới /kaggle/input (đường dẫn có thể
      lồng nhiều cấp, vd /kaggle/input/notebooks/<owner>/<notebook>/checkpoints/...; tìm bằng glob)
      (output lần chạy trước được Add Input); khôi phục toàn bộ trạng thái rồi chạy tiếp từ step đã lưu.
  [ ] An toàn số: kiểm tra torch.isfinite(loss) mỗi bước; NaN -> bỏ bước, ghi log; lặp nhiều lần -> dừng.
  [ ] Test resume (N5): chạy 300 bước, ngắt ~bước 150, resume -> đường loss liền mạch, gần trùng lượt không ngắt.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from data.loader import build_data
from models import build_model
from utils import (CSVLogger, apply_overrides, cosine_lr, get_device, gpu_name, load_config, peak_mem_mb,
                   run_name_of, set_seed, sync, to_ns)

VAL_SEED = 12_345  # cố định -> mọi lượt chạy cùng L đánh giá trên cùng các cửa sổ val
KAGGLE_WORKING = Path("/kaggle/working")


def out_dir(p: str) -> Path:
    """Trên Kaggle, đường dẫn tương đối được đặt dưới /kaggle/working (ngoài thư mục clone, vốn bị xoá mỗi lần
    chạy lại notebook setup) để checkpoint/log được giữ lại khi Save Version."""
    p = Path(p)
    return KAGGLE_WORKING / p if KAGGLE_WORKING.exists() and not p.is_absolute() else p


def save_checkpoint(path: Path, state: dict) -> None:
    # TODO(ĐG): ghi file tạm + os.replace; state đủ khóa theo mục 3.4.
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def find_resume_checkpoint(run_name: str, checkpoint_dir: Path) -> Path | None:
    # TODO(ĐG): tìm <checkpoint_dir>/<run_name>/ckpt_latest.pt và /kaggle/input/**/checkpoints/<run_name>/ckpt_latest.pt
    return None


@torch.no_grad()
def evaluate_bpc(model, data, split, batch_size, seq_len, n_batches, device, amp_ctx) -> float:
    """BPC = loss trung bình (nats) / ln 2, trên n_batches cửa sổ cố định."""
    model.eval()
    losses = []
    for i in range(n_batches):
        x, y = data.get_batch(split, batch_size, seq_len, step=i, seed=VAL_SEED)
        x, y = x.to(device), y.to(device)
        with amp_ctx():
            logits = model(x)
        losses.append(F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)).item())
    model.train()
    return sum(losses) / len(losses) / math.log(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="ghi đè, vd train.lr=3e-4")
    args = ap.parse_args()

    cfg_dict = apply_overrides(load_config(args.config), args.set)
    run_name = cfg_dict["run_name"] = run_name_of(cfg_dict)
    cfg = to_ns(cfg_dict)
    tc, L = cfg.train, cfg.data.seq_len

    micro_bs = tc.batch_chars // (L * tc.grad_accum)
    assert micro_bs >= 1, f"batch_chars={tc.batch_chars} quá nhỏ cho L={L}, grad_accum={tc.grad_accum}"
    chars_per_step = micro_bs * L * tc.grad_accum
    max_steps = tc.train_chars // chars_per_step
    warmup = max(1, int(tc.warmup_frac * max_steps))

    set_seed(cfg.seed)
    device = get_device()
    data = build_data(cfg.data)
    model = build_model(cfg.model, len(data.vocab)).to(device)
    print(f"[{run_name}] model={cfg.model.name} params={model.num_params()/1e6:.2f}M vocab={len(data.vocab)} "
          f"L={L} B={micro_bs}x{tc.grad_accum} steps={max_steps} device={gpu_name(device)}")

    use_fp16 = tc.precision == "fp16" and device.type == "cuda"
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, betas=tuple(tc.betas), weight_decay=tc.weight_decay)

    ckpt_dir, log_dir = out_dir(cfg.paths.checkpoint_dir), out_dir(cfg.paths.log_dir)
    ckpt_path = ckpt_dir / run_name / "ckpt_latest.pt"
    start_step = 0
    resume_from = find_resume_checkpoint(run_name, ckpt_dir)
    if resume_from is not None:
        raise NotImplementedError("TODO(ĐG): khôi phục trạng thái từ checkpoint")
    logger = CSVLogger(log_dir / f"{run_name}.csv", append=start_step > 0)
    gname = gpu_name(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_start = t_last = t_ckpt = time.perf_counter()
    tps_hist, best_val = [], math.inf
    for step in range(start_step, max_steps):
        lr = cosine_lr(step, tc.lr, warmup, max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for micro in range(tc.grad_accum):
            x, y = data.get_batch("train", micro_bs, L, step=step * tc.grad_accum + micro, seed=cfg.seed)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with amp_ctx():
                logits = model(x)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)) / tc.grad_accum
            scaler.scale(loss).backward()
            loss_sum += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(opt)
        scaler.update()

        row = {}
        if (step + 1) % tc.log_every == 0:
            sync(device)
            now = time.perf_counter()
            tps = chars_per_step * tc.log_every / (now - t_last)
            t_last = now
            if step + 1 > tc.log_every:  # bỏ khoảng log đầu (khởi động CUDA) khỏi trung bình
                tps_hist.append(tps)
            row.update(train_loss=f"{loss_sum:.4f}", tokens_per_sec=f"{tps:.0f}")
            print(f"step {step+1:>6} | loss {loss_sum:.4f} | lr {lr:.2e} | {tps:,.0f} char/s")
        if (step + 1) % tc.eval_every == 0 or step + 1 == max_steps:
            val_bpc = evaluate_bpc(model, data, "val", micro_bs, L, tc.eval_batches, device, amp_ctx)
            best_val = min(best_val, val_bpc)
            row["val_bpc"] = f"{val_bpc:.4f}"
            print(f"step {step+1:>6} | VAL bpc {val_bpc:.4f}")
        if row:
            pm = peak_mem_mb(device)
            logger.log(step=step + 1, lr=f"{lr:.3e}", peak_mem_mb="" if math.isnan(pm) else f"{pm:.0f}",
                       gpu_name=gname, time=f"{time.perf_counter() - t_start:.1f}", **row)

        if time.perf_counter() - t_ckpt > tc.ckpt_every_min * 60 or step + 1 == max_steps:
            save_checkpoint(ckpt_path, {"model": model.state_dict(), "step": step + 1, "config": cfg_dict,
                                        "vocab": data.vocab.itos})
            t_ckpt = time.perf_counter()

    pm = peak_mem_mb(device)
    summary = {
        "run_name": run_name, "model": cfg.model.name, "params": model.num_params(), "seq_len": L,
        "steps": max_steps, "train_chars": max_steps * chars_per_step, "best_val_bpc": round(best_val, 4),
        "mean_tokens_per_sec": round(sum(tps_hist) / len(tps_hist)) if tps_hist else None,
        "peak_mem_mb": None if math.isnan(pm) else round(pm, 1),
        "train_time_s": round(time.perf_counter() - t_start, 1), "gpu_name": gname, "config": cfg_dict,
    }
    out = log_dir / f"{run_name}_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
