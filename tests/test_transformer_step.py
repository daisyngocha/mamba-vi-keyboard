"""Riêng TransformerLM (issue #23): KV cache, RoPE theo vị trí tuyệt đối, prefill nhanh, ngoại suy độ dài.
Bổ sung cho tests/test_step.py (test chung step == forward cho mọi model).
"""
import pytest
import torch

from models.base import BaseCharLM
from tests.test_models import DEVICE, VOCAB
from tests.test_step import _build_eval


def _close(a, b):
    torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


def _close_state(s1, s2):
    assert s1["pos"] == s2["pos"] and len(s1["kv"]) == len(s2["kv"])
    for (k1, v1), (k2, v2) in zip(s1["kv"], s2["kv"]):
        _close(k1, k2)
        _close(v1, v2)


@pytest.mark.parametrize("T", [1, 5, 100])
def test_fast_prefill_equals_step_loop(T):
    """prefill nhanh (1 lần forward) = BaseCharLM.prefill (lặp step) cả logits lẫn KV cache."""
    model = _build_eval("transformer")
    x = torch.randint(VOCAB, (2, T), device=DEVICE)
    with torch.no_grad():
        fast_logits, fast_state = model.prefill(x)
        loop_logits, loop_state = BaseCharLM.prefill(model, x)
    _close(fast_logits, loop_logits)
    _close_state(fast_state, loop_state)


def test_prefill_in_two_parts():
    """Nạp 60 ký tự rồi nạp tiếp 40 ký tự vào cache đó = nạp 100 ký tự một lần (RoPE nối tiếp đúng vị trí)."""
    model = _build_eval("transformer")
    x = torch.randint(VOCAB, (2, 100), device=DEVICE)
    with torch.no_grad():
        _, s = model.prefill(x[:, :60])
        logits_a, s_a = model.prefill(x[:, 60:], s)
        logits_b, s_b = model.prefill(x)
    _close(logits_a, logits_b)
    _close_state(s_a, s_b)


def test_longer_than_training_length():
    """Ngoại suy độ dài: chuỗi 600 ký tự (dài hơn L = 256 khi huấn luyện lượt ngắn nhất) — step khớp forward ở cuối chuỗi."""
    model = _build_eval("transformer")
    x = torch.randint(VOCAB, (1, 600), device=DEVICE)
    with torch.no_grad():
        full = model(x)
        _, state = model.prefill(x[:, :590])
        for t in range(590, 600):
            logits, state = model.step(x[:, t], state)
            _close(logits, full[:, t])


def test_step_does_not_modify_input_state():
    """step() trả cache mới dài thêm 1; cache cũ giữ nguyên để beam search / KSR quay lại được."""
    model = _build_eval("transformer")
    x = torch.randint(VOCAB, (2, 20), device=DEVICE)
    with torch.no_grad():
        _, state = model.prefill(x)
        before = [(k.clone(), v.clone()) for k, v in state["kv"]]
        _, new = model.step(x[:, 0], state)
    assert state["pos"] == 20 and new["pos"] == 21
    for (k, v), (k0, v0), (k1, _) in zip(state["kv"], before, new["kv"]):
        assert torch.equal(k, k0) and torch.equal(v, v0)
        assert k1.shape[2] == 21


def test_kv_cache_size_formula():
    """KV cache = 2 · n_layer · T · d_model · byte, tăng tuyến tính theo T (eval_spec mục 5)."""
    model = _build_eval("transformer")
    cfg = model.cfg
    with torch.no_grad():
        for T in (16, 200):
            _, state = model.prefill(torch.randint(VOCAB, (1, T), device=DEVICE))
            itemsize = state["kv"][0][0].element_size()
            assert model.state_nbytes(state) == 2 * cfg.n_layer * T * cfg.d_model * itemsize
    print(f"KV cache: {2 * cfg.n_layer * cfg.d_model * itemsize:,} byte/ký tự ({next(model.parameters()).dtype})")
