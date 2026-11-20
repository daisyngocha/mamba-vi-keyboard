"""Resume: dừng giữa chừng rồi chạy tiếp phải cho đúng kết quả như lượt không bị ngắt (CPU, fp32)."""
import csv
import subprocess
import sys

import torch


def _run(tmp, run_name, *extra):
    cmd = [sys.executable, "train.py", "--config", "configs/smoke.yaml",
           "--set", f"paths.checkpoint_dir={tmp}/ckpt", "--set", f"paths.log_dir={tmp}/logs",
           "--set", f"run_name={run_name}", "--set", "train.log_every=5", *extra]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def _rows(tmp, run_name):
    with open(f"{tmp}/logs/{run_name}.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_resume_matches_uninterrupted(tmp_path):
    _run(tmp_path, "full")
    _run(tmp_path, "cut", "--stop_after", "25")
    out = _run(tmp_path, "cut")
    assert "Resume từ" in out and "step 25/60" in out
    full, cut = _rows(tmp_path, "full"), _rows(tmp_path, "cut")
    assert [r["step"] for r in cut] == [r["step"] for r in full]  # không trùng / thiếu dòng
    for a, b in zip(full, cut):
        assert a["train_loss"] == b["train_loss"] and a["val_bpc"] == b["val_bpc"], (a, b)
    ck = torch.load(tmp_path / "ckpt" / "cut" / "ckpt_latest.pt", weights_only=False)
    assert {"model", "optimizer", "scheduler", "scaler", "step", "rng_state", "config"} <= ck.keys()
    assert ck["step"] == 60


def test_resume_rejects_changed_config(tmp_path):
    _run(tmp_path, "r", "--stop_after", "10")
    cmd = [sys.executable, "train.py", "--config", "configs/smoke.yaml", "--set", f"paths.checkpoint_dir={tmp_path}/ckpt",
           "--set", f"paths.log_dir={tmp_path}/logs", "--set", "run_name=r", "--set", "train.lr=5e-4"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode != 0 and "cấu hình khác" in out.stderr
