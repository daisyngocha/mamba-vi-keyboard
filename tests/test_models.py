"""Kiểm tra giao diện model (mục 3.3). Chạy: pytest -q tests/
Model chưa cài đặt (NotImplementedError) hoặc thiếu mamba-ssm sẽ được skip, không fail.
"""
import pytest
import torch

from models import build_model
from utils import load_config, to_ns

VOCAB = 300
CONFIGS = {"dummy": "configs/smoke.yaml", "mamba": "configs/mamba.yaml", "transformer": "configs/transformer.yaml"}


def _build(name):
    cfg = to_ns(load_config(CONFIGS[name]))
    try:
        return build_model(cfg.model, VOCAB)
    except (NotImplementedError, ImportError) as e:
        pytest.skip(f"{name}: {e}")


@pytest.mark.parametrize("name", list(CONFIGS))
def test_shape_and_num_params(name):
    model = _build(name)
    x = torch.randint(VOCAB, (2, 32))
    assert model(x).shape == (2, 32, VOCAB)
    assert model.num_params() > 0


@pytest.mark.parametrize("name", list(CONFIGS))
def test_causal(name):
    """Đổi token ở vị trí t -> logits ở các vị trí < t phải giữ nguyên."""
    model = _build(name).eval()
    x = torch.randint(VOCAB, (1, 64))
    t = 40
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % VOCAB
    with torch.no_grad():
        a, b = model(x), model(x2)
    torch.testing.assert_close(a[:, :t], b[:, :t], atol=1e-4, rtol=1e-4)


def test_param_balance():
    """Hai model ≈ 8M tham số, lệch nhau ≤ 5% (mục 6–7)."""
    m, t = _build("mamba"), _build("transformer")
    pm, pt = m.num_params(), t.num_params()
    assert abs(pm - pt) / pt <= 0.05, f"mamba={pm:,} transformer={pt:,}"
