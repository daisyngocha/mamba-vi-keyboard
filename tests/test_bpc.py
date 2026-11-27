"""evaluate/bpc.py — BPC toàn bộ test (eval_spec mục 1). Chạy trên CPU hoặc GPU, không cần dữ liệu thật."""
import json
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from evaluate.bpc import bpc_full, n_windows, run_bpc
from evaluate.common import write_eval
from tests.test_suggest import ITOS, NAMES, _tiny

L = 16


def _ids(n, seed=0):
    return np.random.default_rng(seed).integers(0, len(ITOS), n).astype(np.uint16)


def test_n_windows():
    assert n_windows(16 * 5 + 1, 16) == 5          # cần 1 id thêm làm nhãn cho vị trí cuối
    assert n_windows(16 * 5, 16) == 4
    assert n_windows(10, 16) == 0


@pytest.mark.parametrize("name", NAMES)
def test_bpc_matches_manual(name):
    """Mỗi ký tự (trừ ký tự đầu file) được chấm đúng 1 lần, theo đúng ngữ cảnh trong cửa sổ của nó."""
    model = _tiny(name)
    ids = _ids(16 * 7 + 5)                          # 7 cửa sổ, dư 4 ký tự bị bỏ
    res = bpc_full(model, ids, L, dtype="fp32", batch_size=3)
    total = 0.0
    with torch.no_grad():
        for i in range(7):
            x = torch.tensor(ids[i * L:(i + 1) * L], dtype=torch.long)[None].to(next(model.parameters()).device)
            y = torch.tensor(ids[i * L + 1:(i + 1) * L + 1], dtype=torch.long)[None].to(x.device)
            total += F.cross_entropy(model(x)[0].float(), y[0], reduction="sum").item()
    assert res["windows"] == 7 and res["chars"] == 7 * L and res["dropped_tail"] == 4
    assert math.isclose(res["bpc"], total / (7 * L) / math.log(2), rel_tol=1e-4)
    # cửa sổ cùng độ dài -> BPC tổng = trung bình BPC từng cửa sổ (cơ sở cho kiểm định t ghép cặp)
    assert len(res["per_window"]) == 7
    assert math.isclose(res["bpc"], float(np.mean(res["per_window"])), rel_tol=1e-4)


@pytest.mark.parametrize("name", NAMES)
def test_bpc_batch_invariant(name):
    model = _tiny(name)
    ids = _ids(16 * 9 + 1, seed=3)
    a = bpc_full(model, ids, L, dtype="fp32", batch_size=1)["bpc"]
    b = bpc_full(model, ids, L, dtype="fp32", batch_size=9)["bpc"]
    assert math.isclose(a, b, rel_tol=1e-4)


def test_bpc_random_model_near_uniform():
    """Model chưa train ≈ phân bố đều -> BPC ≈ log2(V); sai lệch lớn là dấu hiệu lệch nhãn/đầu vào."""
    res = bpc_full(_tiny("transformer"), _ids(16 * 20 + 1), L, dtype="fp32")
    assert abs(res["bpc"] - math.log2(len(ITOS))) < 0.5


def test_run_bpc_writes_eval(tmp_path):
    (tmp_path / "data").mkdir()
    _ids(16 * 4 + 1).tofile(tmp_path / "data" / "test.bin")
    info = {"step": 10, "seq_len": L}
    res = run_bpc(_tiny("mamba"), "tiny_L16_s0", info, tmp_path / "data", L, "fp32", tmp_path, write_eval)
    run_bpc(_tiny("mamba"), "tiny_L16_s0", info, tmp_path / "data", 2 * L, "fp32", tmp_path, write_eval)
    e = json.loads((tmp_path / "tiny_L16_s0_eval.json").read_text(encoding="utf-8"))
    assert e["bpc"]["bpc"] == res["bpc"] and e["bpc"]["windows"] == 4 and e["bpc"]["checkpoint_step"] == 10
    assert e["bpc_ctx32"]["seq_len"] == 32           # đổi l_ctx -> khóa riêng, không ghi đè kết quả chính
