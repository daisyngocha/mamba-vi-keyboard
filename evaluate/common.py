"""Hàm dùng chung cho evaluate/: nạp checkpoint thành model + vocab, ngữ cảnh gõ phím, ghi file kết quả.
Dùng cho suggest.py, ksr.py (ĐG) và latency.py, BPC (tuần 3).
"""
from __future__ import annotations

import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from data.loader import DOC_ID, CharVocab, normalize
from models import build_model
from utils import to_ns

EVAL_LOG_DIR = Path("logs")


def is_letter(c: str) -> bool:
    """Chữ cái theo eval_spec mục 0 (str.isalpha); <unk>, <doc> không phải chữ cái."""
    return len(c) == 1 and c.isalpha()


def find_checkpoint(run_name: str, checkpoint_dir: str | Path = "checkpoints") -> Path | None:
    """checkpoints/<run_name>/ckpt_latest.pt tại máy, hoặc trong output notebook đã Add Input trên Kaggle."""
    from train import find_resume_checkpoint  # dùng lại đúng cách train.py tìm checkpoint
    return find_resume_checkpoint(run_name, Path(checkpoint_dir))


def load_run(ckpt_path: str | Path, device: torch.device | None = None):
    """Nạp checkpoint do train.py lưu -> (model.eval(), vocab, info). Mamba tự dùng backend torch nếu không có GPU."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)  # file do chính nhóm tạo
    cfg = ck["config"]
    cfg_model = to_ns(dict(cfg["model"]))
    if cfg_model.name == "mamba":
        has_cuda_kernel = device.type == "cuda" and importlib.util.find_spec("mamba_ssm") is not None
        cfg_model.backend = "cuda" if has_cuda_kernel else "torch"
    vocab = CharVocab(ck["vocab"])
    model = build_model(cfg_model, len(vocab))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    info = {"run_name": cfg["run_name"], "model": cfg_model.name, "seq_len": cfg["data"]["seq_len"],
            "seed": cfg["seed"], "step": ck["step"], "checkpoint": str(ckpt_path),
            "backend": getattr(cfg_model, "backend", None)}
    return model, vocab, info


def autocast_ctx(device: torch.device, dtype: str):
    """fp16 bằng autocast (mặc định khi đánh giá, eval_spec mục 0) hoặc fp32 — trọng số giữ fp32 như lúc huấn luyện.
    State sinh từng bước (conv/ssm state của Mamba, KV cache) theo dtype autocast (models/base.compute_dtype)."""
    if dtype == "fp16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


class TypingContext:
    """Văn bản đã gõ + (logits, state) của model trên tối đa l_ctx ký tự cuối (eval_spec mục 0).

    Khi toàn bộ văn bản còn vừa l_ctx: cập nhật tăng dần bằng step() — mỗi ký tự gõ thêm chỉ tốn 1 bước.
    Khi đã dài hơn l_ctx: cửa sổ trượt nên state cũ không dùng lại được -> prefill lại l_ctx ký tự cuối.
    Văn bản rỗng -> ngữ cảnh là <doc> (đầu bài viết), như lúc huấn luyện.
    """

    def __init__(self, model, vocab: CharVocab, l_ctx: int, text: str = "", ids: list[int] | None = None):
        """ids: ngữ cảnh có sẵn dạng id (vd lấy thẳng từ test.bin, có thể chứa <doc>); text: gõ thêm sau đó."""
        self.model, self.vocab, self.l_ctx = model, vocab, l_ctx
        self.device = next(model.parameters()).device
        self.ids: list[int] = list(ids or [])
        self._cache = None          # (logits (1, V), state, số id đã nạp) — chỉ hợp lệ khi len(ids) ≤ l_ctx
        self.push(text)

    def push(self, text: str) -> None:
        """Gõ thêm text (cập nhật state tăng dần nếu còn trong cửa sổ l_ctx)."""
        self.ids.extend(self.vocab.encode(text))
        if self._cache is None or len(self.ids) > self.l_ctx:
            self._cache = None
            return
        logits, state, n = self._cache
        with torch.no_grad():
            for t in self.ids[n:]:
                logits, state = self.model.step(torch.tensor([t], device=self.device), state)
        self._cache = (logits, state, len(self.ids))

    def current(self):
        """(logits vị trí cuối (1, V), state) cho văn bản đã gõ."""
        if self._cache is not None:
            return self._cache[0], self._cache[1]
        window = self.ids[-self.l_ctx:] if self.ids else [DOC_ID]
        x = torch.tensor([window], device=self.device)
        with torch.no_grad():
            logits, state = self.model.prefill(x)
        if 0 < len(self.ids) <= self.l_ctx:
            self._cache = (logits, state, len(self.ids))
        return logits, state


def write_eval(run_name: str, key: str, value: dict, log_dir: Path = EVAL_LOG_DIR) -> Path:
    """Gộp kết quả vào logs/<run_name>_eval.json (mỗi script ghi một khóa: topk, ksr, bpc, latency...)."""
    path = Path(log_dir) / f"{run_name}_eval.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"run_name": run_name}
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


__all__ = ["is_letter", "find_checkpoint", "load_run", "autocast_ctx", "TypingContext", "write_eval", "normalize"]
