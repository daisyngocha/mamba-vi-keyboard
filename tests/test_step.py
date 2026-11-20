"""Giao diện sinh từng bước (issue #22 MB, #23 TF): step() phải cho cùng logits với forward.
Model chưa cài init_state/core_step (NotImplementedError) được skip, không fail.
"""
import pytest
import torch

from models import build_model
from tests.test_models import DEVICE, VOCAB, _cfg

NAMES = ["dummy", "mamba", "transformer"]
T = 128


def _build_eval(name):
    torch.manual_seed(0)
    model = build_model(_cfg(name), VOCAB).to(DEVICE).float().eval()
    try:
        model.init_state(1, DEVICE)
    except NotImplementedError as e:
        pytest.skip(str(e))
    return model


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("batch", [1, 2])
def test_step_matches_forward(name, batch):
    model = _build_eval(name)
    x = torch.randint(VOCAB, (batch, T), device=DEVICE)
    with torch.no_grad():
        full = model(x)
        state = model.init_state(batch, DEVICE)
        for t in range(T):
            logits, state = model.step(x[:, t], state)
            torch.testing.assert_close(logits, full[:, t], atol=1e-3, rtol=1e-3,
                                       msg=lambda m: f"vị trí t={t}: {m}")


@pytest.mark.parametrize("name", NAMES)
def test_prefill_then_step(name):
    """prefill(x[:, :T-1]) rồi step(x[:, T-1]) = forward ở vị trí cuối; prefill trả logits vị trí T-2."""
    model = _build_eval(name)
    x = torch.randint(VOCAB, (2, T), device=DEVICE)
    with torch.no_grad():
        full = model(x)
        logits_pre, state = model.prefill(x[:, :-1])
        logits_last, state = model.step(x[:, -1], state)
    torch.testing.assert_close(logits_pre, full[:, -2], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(logits_last, full[:, -1], atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("name", NAMES)
def test_state_nbytes(name):
    """In kích thước trạng thái theo độ dài đã nạp (RQ3): Mamba kỳ vọng không đổi, Transformer tăng tuyến tính."""
    model = _build_eval(name)
    sizes = {}
    with torch.no_grad():
        for n in (16, 64):
            _, state = model.prefill(torch.randint(VOCAB, (1, n), device=DEVICE))
            sizes[n] = model.state_nbytes(state)
    assert all(v >= 0 for v in sizes.values())
    print(f"{name}: state_nbytes {sizes}")
    if name == "mamba":
        assert sizes[16] == sizes[64] > 0, "trạng thái Mamba phải có kích thước cố định"
    if name == "transformer":
        assert sizes[64] == 4 * sizes[16] > 0, "KV cache phải tăng tuyến tính theo số ký tự"
