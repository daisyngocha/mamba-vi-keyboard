"""Vòng huấn luyện dùng chung cho mọi model. Owner: ĐG.

Chạy:
    python train.py --config configs/smoke.yaml
    python train.py --config configs/mamba.yaml --set data.seq_len=1024 --set seed=1
    python train.py --config configs/transformer.yaml --stop_after 150     # giả lập bị ngắt (test resume)

Đầu ra (mục 3.4):
    <checkpoint_dir>/<run_name>/ckpt_latest.pt     (không commit; trên Kaggle tự đặt dưới /kaggle/working)
    <log_dir>/<run_name>.csv                       cột: step, train_loss, val_bpc, lr, tokens_per_sec,
                                                   peak_mem_mb, gpu_name, time
    <log_dir>/<run_name>_summary.json

Vòng lặp: get_batch -> forward -> cross-entropy -> backward -> clip 1.0 -> AdamW (không weight decay cho norm, bias,
A_log, D), lr cosine có warmup, fp16 autocast + GradScaler (hoặc fp32), batch tính theo ký tự, val trên tập batch cố
định, BPC = loss (nats) / ln 2.

Checkpoint và resume (Hướng dẫn mục 8):
- Lưu mỗi train.ckpt_every_min phút và ở bước cuối: model, optimizer, scheduler, scaler, step, rng_state, config
  (+ vocab, best_val, thời gian đã chạy). Ghi ra file tạm rồi os.replace -> không hỏng file khi bị ngắt giữa lúc ghi.
- Khi khởi động tự tìm ckpt_latest.pt của run_name trong checkpoint_dir và dưới /kaggle/input (output lần chạy trước
  được Add Input), lấy bản có step lớn nhất, khôi phục toàn bộ trạng thái và chạy tiếp. Log CSV của lần trước được
  chép sang và cắt bỏ các dòng sau step của checkpoint. --no_resume để chạy lại từ đầu.
- Batch chỉ phụ thuộc (seed, step) nên chạy tiếp cho ra đúng chuỗi batch như lượt không bị ngắt.
An toàn số: loss không hữu hạn -> bỏ bước (không cập nhật), ghi log; quá train.max_nan_skips lần liên tiếp -> dừng.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.loader import build_data
from models import build_model
from utils import (KAGGLE_INPUT, CSVLogger, apply_overrides, cosine_lr, get_device, gpu_name, load_config,
                   peak_mem_mb, run_name_of, set_seed, sync, to_ns)

VAL_SEED = 12_345  # cố định -> mọi lượt chạy cùng L đánh giá trên cùng các cửa sổ val
KAGGLE_WORKING = Path("/kaggle/working")
CKPT_NAME = "ckpt_latest.pt"


def out_dir(p: str) -> Path:
    """Trên Kaggle, đường dẫn tương đối được đặt dưới /kaggle/working (ngoài thư mục clone, vốn bị xoá mỗi lần
    chạy lại notebook setup) để checkpoint/log được giữ lại khi Save Version."""
    p = Path(p)
    return KAGGLE_WORKING / p if KAGGLE_WORKING.exists() and not p.is_absolute() else p


# ---------------------------------------------------------------- optimizer
def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay cho ma trận trọng số (gồm embedding); không cho norm, bias và A_log, D của Mamba."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.dim() < 2 or getattr(p, "_no_weight_decay", False) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


# ---------------------------------------------------------------- checkpoint
def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available() and len(s["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(s["cuda"])


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)  # file do chính nhóm tạo


def find_resume_checkpoint(run_name: str, checkpoint_dir: Path) -> Path | None:
    """ckpt_latest.pt của run_name trong checkpoint_dir và dưới /kaggle/input; chọn bản có step lớn nhất."""
    cands = [checkpoint_dir / run_name / CKPT_NAME]
    if KAGGLE_INPUT.exists():
        cands += sorted(KAGGLE_INPUT.glob(f"**/checkpoints/{run_name}/{CKPT_NAME}"))
    best, best_step = None, -1
    for c in cands:
        if c.is_file():
            try:
                step = load_checkpoint(c)["step"]
            except Exception as e:  # file hỏng/ghi dở -> bỏ qua
                print(f"Bỏ qua checkpoint lỗi {c}: {e!r}")
                continue
            if step > best_step:
                best, best_step = c, step
    return best


def resume_signature(cfg: dict) -> dict:
    """Các tham số phải giữ nguyên khi chạy tiếp, nếu không kết quả không còn là một lượt chạy."""
    t = cfg["train"]
    return {"model": cfg["model"], "seq_len": cfg["data"]["seq_len"], "train_file": cfg["data"]["train_file"],
            "source": cfg["data"]["source"], "seed": cfg["seed"],
            **{k: t[k] for k in ("batch_chars", "grad_accum", "train_chars", "lr", "warmup_frac", "precision",
                                 "weight_decay", "betas")}}


def trim_csv(path: Path, max_step: int) -> None:
    """Bỏ các dòng log có step > max_step (ghi sau checkpoint cuối, sẽ được chạy lại)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSVLogger.FIELDS)
        w.writeheader()
        w.writerows(r for r in rows if int(r["step"]) <= max_step)


# ---------------------------------------------------------------- đánh giá
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
    ap.add_argument("--no_resume", action="store_true", help="bỏ qua checkpoint có sẵn, chạy từ đầu")
    ap.add_argument("--stop_after", type=int, default=None, help="dừng sau bước này (lưu checkpoint) — test resume")
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
    max_nan_skips = getattr(tc, "max_nan_skips", 20)

    set_seed(cfg.seed)
    device = get_device()
    data = build_data(cfg.data)
    model = build_model(cfg.model, len(data.vocab)).to(device)
    print(f"[{run_name}] model={cfg.model.name} params={model.num_params()/1e6:.2f}M vocab={len(data.vocab)} "
          f"L={L} B={micro_bs}x{tc.grad_accum} steps={max_steps} device={gpu_name(device)}")

    use_fp16 = tc.precision == "fp16" and device.type == "cuda"
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_fp16 else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    opt = torch.optim.AdamW(param_groups(model, tc.weight_decay), lr=tc.lr, betas=tuple(tc.betas))

    ckpt_dir, log_dir = out_dir(cfg.paths.checkpoint_dir), out_dir(cfg.paths.log_dir)
    ckpt_path = ckpt_dir / run_name / CKPT_NAME
    log_path = log_dir / f"{run_name}.csv"
    start_step, best_val, elapsed_before = 0, math.inf, 0.0
    resume_from = None if args.no_resume else find_resume_checkpoint(run_name, ckpt_dir)
    if resume_from is not None:
        ck = load_checkpoint(resume_from)
        if resume_signature(ck["config"]) != resume_signature(cfg_dict):
            sys.exit(f"Checkpoint {resume_from} có cấu hình khác lượt chạy hiện tại — dùng --no_resume hoặc "
                     f"đặt run_name khác.\n  cũ: {resume_signature(ck['config'])}\n  mới: {resume_signature(cfg_dict)}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        set_rng_state(ck["rng_state"])
        start_step, best_val, elapsed_before = ck["step"], ck.get("best_val", math.inf), ck.get("elapsed_s", 0.0)
        prev_log = resume_from.parent.parent.parent / "logs" / f"{run_name}.csv"
        if not log_path.exists() and prev_log.exists():
            log_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(prev_log, log_path)
        if log_path.exists():
            trim_csv(log_path, start_step)
        print(f"Resume từ {resume_from} tại step {start_step}/{max_steps}")
    elif ckpt_path.exists() and args.no_resume:
        print(f"--no_resume: ghi đè {ckpt_path}")
    if start_step >= max_steps:
        print("Lượt chạy đã hoàn thành trước đó.")
        return

    logger = CSVLogger(log_path, append=start_step > 0)
    gname = gpu_name(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def state(step_done: int) -> dict:
        return {"model": model.state_dict(), "optimizer": opt.state_dict(),
                "scheduler": {"type": "cosine_warmup", "warmup": warmup, "max_steps": max_steps},
                "scaler": scaler.state_dict(), "step": step_done, "rng_state": rng_state(), "config": cfg_dict,
                "vocab": data.vocab.itos, "best_val": best_val,
                "elapsed_s": elapsed_before + time.perf_counter() - t_start}

    t_start = t_last = t_ckpt = time.perf_counter()
    last_log_step, tps_hist, nan_streak = start_step, [], 0
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

        row = {}
        if not math.isfinite(loss_sum):
            nan_streak += 1
            opt.zero_grad(set_to_none=True)
            row["train_loss"] = "nan"
            print(f"step {step+1:>6} | loss không hữu hạn -> bỏ bước ({nan_streak}/{max_nan_skips})")
            if nan_streak >= max_nan_skips:
                logger.log(step=step + 1, lr=f"{lr:.3e}", gpu_name=gname, train_loss="nan",
                           time=f"{elapsed_before + time.perf_counter() - t_start:.1f}")
                sys.exit(f"Dừng: {nan_streak} bước liên tiếp loss NaN/inf — thử precision=fp32 hoặc giảm lr.")
        else:
            nan_streak = 0
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            scaler.step(opt)
            scaler.update()

        stop_now = args.stop_after is not None and step + 1 >= args.stop_after
        if (step + 1) % tc.log_every == 0 or stop_now:  # luôn ghi 1 dòng khi dừng sớm (đo bộ nhớ ngắn)
            sync(device)
            now = time.perf_counter()
            tps = chars_per_step * (step + 1 - last_log_step) / (now - t_last)
            if last_log_step > start_step:  # bỏ khoảng log đầu mỗi phiên (khởi động CUDA) khỏi trung bình
                tps_hist.append(tps)
            t_last, last_log_step = now, step + 1
            row.setdefault("train_loss", f"{loss_sum:.4f}")
            row["tokens_per_sec"] = f"{tps:.0f}"
            print(f"step {step+1:>6} | loss {loss_sum:.4f} | lr {lr:.2e} | {tps:,.0f} char/s")
        if (step + 1) % tc.eval_every == 0 or step + 1 == max_steps:
            val_bpc = evaluate_bpc(model, data, "val", micro_bs, L, tc.eval_batches, device, amp_ctx)
            best_val = min(best_val, val_bpc)
            row["val_bpc"] = f"{val_bpc:.4f}"
            print(f"step {step+1:>6} | VAL bpc {val_bpc:.4f}")
        if row:
            pm = peak_mem_mb(device)
            logger.log(step=step + 1, lr=f"{lr:.3e}", peak_mem_mb="" if math.isnan(pm) else f"{pm:.0f}",
                       gpu_name=gname, time=f"{elapsed_before + time.perf_counter() - t_start:.1f}", **row)

        if time.perf_counter() - t_ckpt > tc.ckpt_every_min * 60 or step + 1 == max_steps or stop_now:
            save_checkpoint(ckpt_path, state(step + 1))
            t_ckpt = time.perf_counter()
        if stop_now and step + 1 < max_steps:
            print(f"--stop_after: dừng ở step {step+1}, checkpoint tại {ckpt_path}")
            return

    pm = peak_mem_mb(device)
    summary = {
        "run_name": run_name, "model": cfg.model.name, "params": model.num_params(), "seq_len": L,
        "steps": max_steps, "train_chars": max_steps * chars_per_step, "best_val_bpc": round(best_val, 4),
        "mean_tokens_per_sec": round(sum(tps_hist) / len(tps_hist)) if tps_hist else None,
        "peak_mem_mb": None if math.isnan(pm) else round(pm, 1),
        "train_time_s": round(elapsed_before + time.perf_counter() - t_start, 1),
        "resumed_from_step": start_step or None, "gpu_name": gname, "config": cfg_dict,
    }
    out = log_dir / f"{run_name}_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
