"""Vòng huấn luyện dùng chung cho mọi model. Owner: lead (khung) + Đặng Duyên Hà (hoàn thiện).

Chạy:
    python -m src.train --config configs/smoke.yaml
    python -m src.train --config configs/mamba.yaml --set train.max_steps=200 --set seed=1

Đầu ra: results/<run_name>/{config.yaml, metrics.csv, summary.json, last.pt}
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from src.dataset import build_dataset
from src.models import build_model
from src.utils import CSVLogger, count_params, cosine_lr, get_device, load_config, peak_mem_mb, set_seed, to_ns


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for item in overrides:
        key, val = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        parsed = yaml.safe_load(val)
        if isinstance(parsed, str):  # PyYAML đọc "3e-4" thành chuỗi
            try:
                parsed = float(parsed)
            except ValueError:
                pass
        node[leaf] = parsed
    return cfg


@torch.no_grad()
def evaluate(model, ds, split, device, n_batches, amp_ctx):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = ds.get_batch(split, device)
        with amp_ctx():
            logits = model(x)
        losses.append(F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], help="ghi đè, vd train.lr=3e-4")
    args = ap.parse_args()

    cfg_dict = apply_overrides(load_config(args.config), args.set)
    cfg = to_ns(cfg_dict)
    run_name = cfg.run_name or f"{cfg.model.name}_s{cfg.seed}_{datetime.now():%m%d_%H%M}"
    out_dir = Path(cfg.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict, allow_unicode=True), encoding="utf-8")

    set_seed(cfg.seed)
    device = get_device()
    ds = build_dataset(cfg.data)
    model = build_model(cfg.model, len(ds.vocab)).to(device)
    n_params = count_params(model)
    print(f"[{run_name}] model={cfg.model.name} params={n_params/1e6:.2f}M vocab={len(ds.vocab)} device={device}")

    # AMP: bf16 nếu GPU hỗ trợ, ngược lại fp16 + GradScaler (T4/P100 trên Kaggle).
    use_amp = cfg.train.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    amp_ctx = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if use_amp else nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, betas=tuple(cfg.train.betas),
                            weight_decay=cfg.train.weight_decay)
    logger = CSVLogger(out_dir / "metrics.csv")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    tokens_per_step = cfg.data.batch_size * cfg.data.seq_len
    t_start = t_last = time.perf_counter()
    tps_hist, best_val = [], math.inf
    for step in range(cfg.train.max_steps):
        lr = cosine_lr(step, cfg.train.lr, cfg.train.warmup_steps, cfg.train.max_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        x, y = ds.get_batch("train", device)
        with amp_ctx():
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(opt)
        scaler.update()

        if (step + 1) % cfg.train.log_every == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            now = time.perf_counter()
            tps = tokens_per_step * cfg.train.log_every / (now - t_last)
            t_last = now
            tps_hist.append(tps)
            logger.log(step=step + 1, split="train", loss=f"{loss.item():.4f}", bpc=f"{loss.item()/math.log(2):.4f}",
                       lr=f"{lr:.2e}", tokens_per_s=f"{tps:.0f}", peak_mem_mb=f"{peak_mem_mb(device):.0f}",
                       elapsed_s=f"{now - t_start:.1f}")
            print(f"step {step+1:>6} | loss {loss.item():.4f} | lr {lr:.2e} | {tps:,.0f} tok/s")

        if (step + 1) % cfg.train.eval_every == 0 or step + 1 == cfg.train.max_steps:
            val = evaluate(model, ds, "val", device, cfg.train.eval_batches, amp_ctx)
            best_val = min(best_val, val)
            logger.log(step=step + 1, split="val", loss=f"{val:.4f}", bpc=f"{val/math.log(2):.4f}",
                       elapsed_s=f"{time.perf_counter() - t_start:.1f}")
            print(f"step {step+1:>6} | VAL loss {val:.4f} | bpc {val/math.log(2):.3f}")

        if (step + 1) % cfg.train.save_every == 0 or step + 1 == cfg.train.max_steps:
            torch.save({"model": model.state_dict(), "config": cfg_dict, "step": step + 1,
                        "vocab": ds.vocab.itos}, out_dir / "last.pt")

    summary = {
        "run_name": run_name, "model": cfg.model.name, "params": n_params, "steps": cfg.train.max_steps,
        "best_val_loss": round(best_val, 4), "best_val_bpc": round(best_val / math.log(2), 4),
        "mean_tokens_per_s": round(sum(tps_hist) / max(1, len(tps_hist))),
        "peak_mem_mb": round(peak_mem_mb(device), 1) if device.type == "cuda" else None, "train_time_s": round(time.perf_counter() - t_start, 1),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
