"""bench.py: đo được trên CPU và ghi OOM thay vì dừng."""
from types import SimpleNamespace as NS

import torch

import bench
from utils import load_config, to_ns

ARGS = NS(warmup=1, repeat=2)


def test_bench_one_ok():
    cfg = to_ns(load_config("configs/transformer.yaml"))
    cfg.model.n_layer, cfg.train.precision = 1, "fp32"
    row = bench.bench_one(cfg, 32, 2, 275, torch.device("cpu"), ARGS)
    assert row["status"] == "ok" and row["train_step_ms"] > 0 and row["chars_per_sec"] > 0


def test_bench_one_oom(monkeypatch):
    def boom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    monkeypatch.setattr(bench, "build_model", boom)
    cfg = to_ns(load_config("configs/transformer.yaml"))
    row = bench.bench_one(cfg, 4096, 8, 275, torch.device("cpu"), ARGS)
    assert row["status"] == "OOM"
