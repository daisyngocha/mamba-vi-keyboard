"""Giao diện sinh từng bước (issue #22 MB, #23 TF): step() phải cho cùng logits với forward;
reorder_state() chọn lại / nhân bản đúng chuỗi cho beam search (issue #26).
Model chưa cài init_state/core_step (NotImplementedError) được skip, không fail.
"""
import pytest
import torch

from models import build_model
from models.base import BaseCharLM
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


@pytest.mark.parametrize("name", NAMES)
def test_reorder_state(name):
    """Nạp 2 chuỗi, chọn lại theo idx = [1, 0, 1] rồi step tiếp = nạp lại từ đầu các chuỗi x[idx]."""
    model = _build_eval(name)
    x = torch.randint(VOCAB, (2, T), device=DEVICE)
    idx = torch.tensor([1, 0, 1], device=DEVICE)
    with torch.no_grad():
        _, state = model.prefill(x[:, :-1])
        got, _ = model.step(x[idx, -1], model.reorder_state(state, idx))
        ref, _ = model.prefill(x[idx])
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)


def test_reorder_state_nested():
    """Không cần model: state lồng nhau (dict / list / tuple / int) được chọn lại theo dim 0, có bản sao riêng."""
    a = torch.arange(6.0).reshape(3, 2)
    state = {"layers": [(a, a * 10)], "pos": 7}
    out = BaseCharLM.reorder_state(state, [2, 2, 0])
    k, v = out["layers"][0]
    assert isinstance(out["layers"], list) and isinstance(out["layers"][0], tuple)
    torch.testing.assert_close(k, a[[2, 2, 0]])
    torch.testing.assert_close(v, a[[2, 2, 0]] * 10)
    assert out["pos"] == 7
    k[0].fill_(-1)  # cập nhật tại chỗ một nhánh không được làm đổi nhánh nhân bản cùng nguồn hay state gốc
    assert k[1, 0] == 4 and a[2, 0] == 4
