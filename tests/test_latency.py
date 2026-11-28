"""evaluate/latency.py (RQ3, eval_spec mục 5) — model nhỏ, chạy trên DEVICE (CPU hoặc GPU)."""
import math

import numpy as np
import pytest

from evaluate.latency import letters_tail, measure_T, state_formula_bytes
from tests.test_suggest import ITOS, NAMES, VOCAB, _tiny


def _ids(n=600, seed=0):
    return np.random.default_rng(seed).integers(0, len(ITOS), n).astype(np.uint16)


def test_letters_tail():
    assert letters_tail(list("ab ba")) == "ba"
    assert letters_tail(list("ab ") + []) == ""
    assert letters_tail(["a", "<doc>"]) == ""


@pytest.mark.parametrize("name", NAMES)
def test_measure_T_fields_and_state(name):
    model = _tiny(name)
    ids = _ids()
    r16 = measure_T(model, VOCAB, ids, 16, "fp32", start=200, warmup=1, n_keys=5)
    r64 = measure_T(model, VOCAB, ids, 64, "fp32", start=200, warmup=1, n_keys=5)
    for r in (r16, r64):
        assert r["n_keys"] == 5 and r["step_ms"] > 0 and r["suggest_ms"] > 0
        assert math.isclose(r["total_ms"], r["step_ms"] + r["suggest_ms"], rel_tol=0.5)   # trung vị tổng ≈ tổng trung vị
        assert r["total_p90_ms"] >= r["total_ms"]
    if name == "mamba":
        assert r16["state_mb"] == r64["state_mb"]                       # state cố định, không phụ thuộc T
    else:
        assert math.isclose(r64["state_mb"] / r16["state_mb"], 4.0)      # KV cache ∝ T


@pytest.mark.parametrize("name", NAMES)
def test_state_formula_matches_measured(name):
    """Công thức eval_spec mục 5 (đổi sang fp32 = 4 byte) khớp state_nbytes thực tế sau khi nạp T ký tự."""
    model = _tiny(name)
    T = 32
    r = measure_T(model, VOCAB, _ids(), T, "fp32", start=200, warmup=1, n_keys=1)
    info = {"model": name}
    want = state_formula_bytes(model, info, T, bytes_per=4) / 2 ** 20
    assert math.isclose(r["state_mb"], want, rel_tol=1e-9)
