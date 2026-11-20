"""Kiểm tra giao diện model (mục 3.3). Chạy: pytest -q tests/
Mamba dùng backend "cuda" (mamba-ssm) khi có GPU + wheel, ngược lại dùng backend "torch" (cùng tham số) để chạy trên CPU.
"""
import importlib.util

import pytest
import torch

from models import build_model
from utils import load_config, to_ns

VOCAB = 275
CONFIGS = {"dummy": "configs/smoke.yaml", "mamba": "configs/mamba.yaml", "transformer": "configs/transformer.yaml"}
HAS_MAMBA_CUDA = torch.cuda.is_available() and importlib.util.find_spec("mamba_ssm") is not None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _cfg(name, **override):
    cfg = to_ns(load_config(CONFIGS[name])).model
    if name == "mamba" and not HAS_MAMBA_CUDA:
        cfg.backend = "torch"
    for k, v in override.items():
        setattr(cfg, k, v)
    return cfg


def _build(name, **override):
    torch.manual_seed(0)
    return build_model(_cfg(name, **override), VOCAB).to(DEVICE)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_shape_and_num_params(name):
    model = _build(name)
    x = torch.randint(VOCAB, (2, 32), device=DEVICE)
    assert model(x).shape == (2, 32, VOCAB)
    assert model.num_params() > 0


@pytest.mark.parametrize("name", list(CONFIGS))
def test_causal(name):
    """Đổi token ở vị trí t -> logits ở các vị trí < t phải giữ nguyên."""
    model = _build(name).eval()
    x = torch.randint(VOCAB, (1, 64), device=DEVICE)
    t = 40
    x2 = x.clone()
    x2[0, t] = (x2[0, t] + 1) % VOCAB
    with torch.no_grad():
        a, b = model(x), model(x2)
    torch.testing.assert_close(a[:, :t], b[:, :t], atol=1e-4, rtol=1e-4)
    assert not torch.allclose(a[:, t:], b[:, t:])  # và vị trí ≥ t thì có thay đổi


def test_param_balance():
    """Hai model ≈ 5–10M tham số, lệch nhau ≤ 5% (mục 6–7)."""
    pm, pt = _build("mamba").num_params(), _build("transformer").num_params()
    assert 5e6 <= pm <= 10e6 and 5e6 <= pt <= 10e6
    assert abs(pm - pt) / pt <= 0.05, f"mamba={pm:,} transformer={pt:,}"


def test_no_weight_decay_on_A_log_D():
    from train import param_groups
    model = _build("mamba")
    no_decay = {id(p) for p in param_groups(model, 0.1)[1]["params"]}
    for name, p in model.named_parameters():
        if name.endswith(("A_log", ".D")) or p.dim() < 2:
            assert id(p) in no_decay, name


@pytest.mark.skipif(not HAS_MAMBA_CUDA, reason="cần GPU + mamba-ssm")
def test_mamba_cuda_matches_torch_backend():
    """Cùng trọng số: kernel CUDA (mamba-ssm) và bản PyTorch tham chiếu cho cùng logits."""
    cuda_m = _build("mamba", backend="cuda", n_layer=2).float().eval()
    ref_m = _build("mamba", backend="torch", n_layer=2).float().eval()
    ref_m.load_state_dict(cuda_m.state_dict())
    x = torch.randint(VOCAB, (2, 128), device=DEVICE)
    with torch.no_grad():
        torch.testing.assert_close(cuda_m(x), ref_m(x), atol=2e-3, rtol=2e-3)


def test_transformer_sdpa_matches_explicit_attention():
    """SDPA nhân quả = attention viết tay với mask tam giác (kiểm tra RoPE + is_causal)."""
    import torch.nn.functional as F
    from models.transformer_lm import apply_rope, rope_cos_sin
    model = _build("transformer", n_layer=1).float().eval()
    attn = model.blocks[0].attn
    h = torch.randn(2, 50, model.cfg.d_model, device=DEVICE)
    B, L, D = h.shape
    q, k, v = attn.qkv(h).view(B, L, 3, attn.n_head, attn.head_dim).permute(2, 0, 3, 1, 4)
    cos, sin = rope_cos_sin(L, attn.head_dim, attn.rope_base, h.device)
    q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    s = (q @ k.transpose(-1, -2)) / attn.head_dim**0.5
    s = s.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool, device=h.device), 1), float("-inf"))
    ref = attn.proj((F.softmax(s, -1) @ v).transpose(1, 2).reshape(B, L, D))
    with torch.no_grad():
        torch.testing.assert_close(attn(h), ref, atol=1e-4, rtol=1e-4)
