"""Tiện ích dùng chung: config, seed, logging, đo bộ nhớ. Owner: lead."""
from __future__ import annotations

import copy
import csv
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml


# ---------- Config ----------
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike) -> dict:
    """Đọc YAML, hỗ trợ khoá `base:` để kế thừa (đường dẫn tương đối theo file hiện tại)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    return cfg


def to_ns(d: dict) -> SimpleNamespace:
    """dict -> namespace để truy cập cfg.model.d_model."""
    return SimpleNamespace(**{k: to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})


# ---------- Reproducibility ----------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- Model helpers ----------
def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_lr(step: int, base_lr: float, warmup: int, max_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------- Đo hiệu năng ----------
def peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    return float("nan")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    def __init__(self, device: torch.device):
        self.device = device

    def __enter__(self):
        sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.elapsed = time.perf_counter() - self.t0


# ---------- Logging ----------
class CSVLogger:
    """Ghi metrics ra CSV. Cột cố định để bảng kết quả trên slide sinh tự động."""

    FIELDS = ["step", "split", "loss", "bpc", "lr", "tokens_per_s", "peak_mem_mb", "elapsed_s"]

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, **row) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow({k: row.get(k, "") for k in self.FIELDS})
