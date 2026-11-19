"""Tiện ích dùng chung: config, seed, lịch lr, đo bộ nhớ/thời gian, log CSV."""
from __future__ import annotations

import copy
import csv
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


# ---------- Config ----------
def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | os.PathLike) -> dict:
    """Đọc YAML, hỗ trợ khóa `base:` để kế thừa (đường dẫn tương đối theo file hiện tại)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = cfg.pop("base", None)
    return _deep_merge(load_config(path.parent / base), cfg) if base else cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """--set train.lr=3e-4 --set data.seq_len=1024"""
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


def to_ns(d: dict) -> SimpleNamespace:
    return SimpleNamespace(**{k: to_ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def run_name_of(cfg: dict) -> str:
    """Quy ước mục 3.4: <model>_L<len>_s<seed>, vd mamba_L1024_s0."""
    return cfg.get("run_name") or f"{cfg['model']['name']}_L{cfg['data']['seq_len']}_s{cfg['seed']}"


# ---------- Reproducibility ----------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_name(device: torch.device) -> str:
    return torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"


# ---------- Lịch learning rate ----------
def cosine_lr(step: int, base_lr: float, warmup: int, max_steps: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


# ---------- Đo hiệu năng ----------
def peak_mem_mb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timer:
    """with Timer(device) as t: ...  -> t.elapsed (giây), có cuda.synchronize hai đầu."""

    def __init__(self, device: torch.device):
        self.device = device

    def __enter__(self):
        sync(self.device)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        sync(self.device)
        self.elapsed = time.perf_counter() - self.t0


# ---------- Log CSV (mục 3.4) ----------
class CSVLogger:
    """Cột cố định để bảng/biểu đồ sinh tự động. time = số giây kể từ đầu lượt chạy."""

    FIELDS = ["step", "train_loss", "val_bpc", "lr", "tokens_per_sec", "peak_mem_mb", "gpu_name", "time"]

    def __init__(self, path: Path, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not (append and self.path.exists()):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, **row) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow({k: row.get(k, "") for k in self.FIELDS})
