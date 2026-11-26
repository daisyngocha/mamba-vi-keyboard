"""Riêng MambaLM (issue #22): prefill nhanh, state không bị sửa tại chỗ, kích thước state cố định.
Bổ sung cho tests/test_step.py (test chung step == forward cho mọi model).
Trên CPU chạy backend torch; trên Kaggle GPU có mamba-ssm thì tự chạy backend cuda (xem tests/test_models.py).
"""
import pytest
import torch

from models.base import BaseCharLM
from tests.test_models import DEVICE, VOCAB
from tests.test_step import _build_eval


def _close(a, b):
    torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)


def _close_state(s1, s2):
    assert len(s1) == len(s2)
    for (c1, h1), (c2, h2) in zip(s1, s2):
        _close(c1.float(), c2.float())
        _close(h1.float(), h2.float())


@pytest.mark.parametrize("T", [2, 4, 100])   # T < d_conv, T = d_conv, T > d_conv
def test_fast_prefill_equals_step_loop(T):
    """prefill nhanh (1 lần forward) = BaseCharLM.prefill (lặp step) cả logits lẫn state."""
    model = _build_eval("mamba")
    x = torch.randint(VOCAB, (2, T), device=DEVICE)
    with torch.no_grad():
        fast_logits, fast_state = model.prefill(x)
        loop_logits, loop_state = BaseCharLM.prefill(model, x)
    _close(fast_logits, loop_logits)
    _close_state(fast_state, loop_state)


def test_prefill_in_two_parts():
    """Nạp 60 ký tự rồi nạp tiếp 40 ký tự vào state đó = nạp 100 ký tự một lần."""
    model = _build_eval("mamba")
    x = torch.randint(VOCAB, (2, 100), device=DEVICE)
    with torch.no_grad():
        _, s = model.prefill(x[:, :60])
        logits_a, s_a = model.prefill(x[:, 60:], s)
        logits_b, s_b = model.prefill(x)
    _close(logits_a, logits_b)
    _close_state(s_a, s_b)


def test_step_does_not_modify_input_state():
    """step() trả state mới; state cũ giữ nguyên để beam search / KSR quay lại được."""
    model = _build_eval("mamba")
    x = torch.randint(VOCAB, (2, 20), device=DEVICE)
    with torch.no_grad():
        _, state = model.prefill(x)
        before = [(c.clone(), h.clone()) for c, h in state]
        model.step(x[:, 0], state)
    for (c, h), (c0, h0) in zip(state, before):
        assert torch.equal(c, c0) and torch.equal(h, h0)


def test_state_shape_and_size():
    """Mỗi block: conv_state (B, d_inner, d_conv), ssm_state (B, d_inner, d_state); tổng byte không phụ thuộc T."""
    model = _build_eval("mamba")
    cfg = model.cfg
    d_inner = cfg.expand * cfg.d_model
    sizes = []
    with torch.no_grad():
        for T in (8, 300):
            _, state = model.prefill(torch.randint(VOCAB, (1, T), device=DEVICE))
            assert len(state) == cfg.n_layer
            for conv, ssm in state:
                assert conv.shape == (1, d_inner, cfg.d_conv)
                assert ssm.shape == (1, d_inner, cfg.d_state)
            sizes.append(model.state_nbytes(state))
    per = sum(c.element_size() * d_inner * cfg.d_conv + h.element_size() * d_inner * cfg.d_state for c, h in state)
    assert sizes[0] == sizes[1] == per
    print(f"Mamba state: {sizes[0]:,} byte (batch 1, {next(model.parameters()).dtype})")
