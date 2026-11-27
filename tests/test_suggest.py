"""evaluate/suggest.py và evaluate/ksr.py (issue #24) — chạy trên CPU, không cần dữ liệu thật.
Model nhỏ khởi tạo ngẫu nhiên + từ vựng 2 chữ cái để so beam search với liệt kê vét cạn mọi từ.
"""
import itertools
import json
import math

import numpy as np
import pytest
import torch

from data.loader import DOC_ID, CharVocab
from evaluate.common import TypingContext, load_run, write_eval
from evaluate.ksr import ksr_passages, run_ksr, run_topk, score_topk, simulate_typing
from evaluate.suggest import split_prefix, suggest, suggest_from_state
from tests.test_models import DEVICE, _cfg

ITOS = ["<unk>", "<doc>", " ", ".", "a", "b"]          # 2 chữ cái: a, b
VOCAB = CharVocab(ITOS)
NAMES = ["mamba", "transformer"]


def _tiny(name):
    torch.manual_seed(1)
    from models import build_model
    over = {"d_model": 32, "n_layer": 2} if name == "transformer" else {"d_model": 32, "n_layer": 2}
    if name == "transformer":
        over["n_head"] = 2
    return build_model(_cfg(name, **over), len(ITOS)).to(DEVICE).eval()   # GPU + mamba-ssm -> backend cuda


def _exhaustive(model, typed, max_len, k=3):
    """Điểm đúng theo eval_spec mục 2 cho MỌI từ trên {a, b} dài ≤ max_len ký tự sinh thêm."""
    prefix = split_prefix(typed)
    letters = [i for i, c in enumerate(ITOS) if c.isalpha() and len(c) == 1]
    scores = {}
    for n in range(0 if prefix else 1, max_len + 1):
        for gen in itertools.product("ab", repeat=n):
            w = prefix + "".join(gen)
            ids = torch.tensor([VOCAB.encode(typed + "".join(gen))], device=DEVICE)
            with torch.no_grad():
                logp = torch.log_softmax(model(ids)[0].float(), -1)
            T0 = len(VOCAB.encode(typed))
            s = sum(logp[T0 - 1 + j, VOCAB.stoi[ch]].item() for j, ch in enumerate(gen))
            term = torch.logsumexp(logp[-1, [i for i in range(len(ITOS)) if i not in letters]], 0).item()
            scores[w] = s + term
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("typed", ["ab ba a", "ab ba ", "b."])
def test_beam_equals_exhaustive(name, typed):
    """Beam đủ rộng (≥ 2^3 nhánh) phải ra đúng top-3 của liệt kê vét cạn — kiểm tra điểm + hạng tử kết thúc."""
    model = _tiny(name)
    got = suggest(model, VOCAB, typed, k=3, beam=8, max_word_len=3, return_scores=True)
    want = _exhaustive(model, typed, 3)
    assert [w for w, _ in got] == [w for w, _ in want]
    for (_, a), (_, b) in zip(got, want):
        assert math.isclose(a, b, abs_tol=1e-3)


@pytest.mark.parametrize("name", NAMES)
def test_modes_and_state_untouched(name):
    model = _tiny(name)
    a = suggest(model, VOCAB, "ab ba ab", max_word_len=4)          # mode a: mọi gợi ý bắt đầu bằng prefix "ab"
    assert len(a) == 3 and all(w.startswith("ab") for w in a) and len(set(a)) == 3
    b = suggest(model, VOCAB, "ab ba ", max_word_len=4)            # mode b: từ mới, ≥ 1 chữ cái
    assert len(b) == 3 and all(w and w.isalpha() for w in b)
    ctx = TypingContext(model, VOCAB, 64, "ab ba ")
    logits, state = ctx.current()
    before = [t.clone() for t in _tensors(state)]
    suggest_from_state(model, VOCAB, logits, state, "")
    assert all(torch.equal(x, y) for x, y in zip(_tensors(state), before))


def _tensors(state):
    if isinstance(state, torch.Tensor):
        return [state]
    if isinstance(state, dict):
        return [t for v in state.values() for t in _tensors(v)]
    if isinstance(state, (list, tuple)):
        return [t for v in state for t in _tensors(v)]
    return []


@pytest.mark.parametrize("name", NAMES)
def test_typing_context_incremental_equals_prefill(name):
    """Gõ tăng dần (step) = prefill lại; vượt l_ctx thì dùng cửa sổ l_ctx ký tự cuối."""
    model = _tiny(name)
    ctx = TypingContext(model, VOCAB, l_ctx=20, text="ab ")
    ctx.current()
    for piece in ["ba", " a", "b."]:
        ctx.push(piece)
        got, _ = ctx.current()
        with torch.no_grad():
            want, _ = model.prefill(torch.tensor([ctx.ids[-20:]], device=DEVICE))
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)
    ctx.push("ab ba ab ba ab ba")                                 # > 20 ký tự -> cửa sổ trượt
    got, _ = ctx.current()
    with torch.no_grad():
        want, _ = model.prefill(torch.tensor([ctx.ids[-20:]], device=DEVICE))
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_simulate_typing_counts():
    text = "ab ba, aab."                                              # 11 ký tự, 3 từ, 3 ký tự không phải chữ cái
    oracle = simulate_typing(text, lambda typed: [_word_at(text, len(typed))])   # gợi ý luôn đúng từ đang gõ
    assert oracle == {"chars": 11, "keys": 3 + 4, "selections": 3}  # mỗi từ 1 chạm + 4 ký tự gõ tay (" ", ",", " ", ".")
    blind = simulate_typing(text, lambda typed: [])
    assert blind["keys"] == 11 and blind["selections"] == 0          # không gợi ý -> KSR = 0
    late = simulate_typing("abba ", lambda typed: ["abba"] if len(typed) >= 2 else [])
    assert late == {"chars": 5, "keys": 2 + 1 + 1, "selections": 1}  # gõ "ab", chạm chọn, gõ " "


def _word_at(text, i):
    s = i
    while s > 0 and text[s - 1].isalpha():
        s -= 1
    e = i
    while e < len(text) and text[e].isalpha():
        e += 1
    return text[s:e]


def test_score_topk():
    recs = [{"mode": "a", "target": "Hà", "suggestions": ["Hà", "hà", "Hải"]},
            {"mode": "a", "target": "Nội", "suggestions": ["nội", "Nam", "Nga"]},
            {"mode": "b", "target": "đẹp", "suggestions": ["mưa", "đẹp", "nắng"]}]
    s = score_topk(recs)
    assert s["a"] == {"n": 2, "top1": 0.5, "top3": 0.5, "top3_nocase": 1.0}
    assert s["b"]["top1"] == 0.0 and s["b"]["top3"] == 1.0 and s["all"]["n"] == 3


def test_ksr_passages_boundaries():
    words = ["ab", "ba", "aab", "bba", "a", "b"]
    rng = np.random.default_rng(0)
    text = ""
    for d in range(5):                                                # 5 bài, mỗi bài mở đầu bằng <doc>
        text += "\x00" + " ".join(rng.choice(words, 400)) + "."
    ids = np.array([DOC_ID if c == "\x00" else VOCAB.stoi[c] for c in text], dtype=np.uint16)
    ps = ksr_passages(ids, VOCAB, n=8, length=60, seed=0, max_context=100)
    assert len(ps) == 8
    spans = sorted((p["start"], p["start"] + len(p["text"])) for p in ps)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))   # không chồng nhau
    for p in ps:
        s, t = p["start"], p["text"]
        assert t[0].isalpha() and not VOCAB.itos[ids[s - 1]].isalpha()                  # bắt đầu ở đầu từ
        nxt = VOCAB.itos[ids[s + len(t)]]
        assert not (t[-1].isalpha() and nxt.isalpha())                                   # không cắt ngang từ
        assert DOC_ID not in p["context_ids"][1:] and len(p["context_ids"]) <= 100        # ngữ cảnh trong cùng bài


@pytest.mark.parametrize("name", NAMES)
def test_end_to_end_checkpoint(tmp_path, name):
    """Lưu checkpoint đúng định dạng train.py -> load_run -> Top-k + KSR -> logs/<run>_eval.json."""
    model = _tiny(name)
    cfg = {"run_name": f"{name}_L64_s0", "seed": 0, "data": {"seq_len": 64},
           "model": {k: v for k, v in vars(model.cfg).items()}}
    path = tmp_path / "ckpt_latest.pt"
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save({"model": sd, "config": cfg, "vocab": ITOS, "step": 123}, path)
    m2, vocab, info = load_run(path, torch.device("cpu"))
    assert info["seq_len"] == 64 and info["step"] == 123
    items = [{"id": 0, "mode": "a", "typed": "ab ba a", "prefix": "a", "target": "ab"},
             {"id": 1, "mode": "b", "typed": "ab ba ", "prefix": "", "target": "ba"}]
    recs, _ = run_topk(m2, vocab, items, l_ctx=64, log_every=0)
    assert len(recs) == 2 and all(len(r["suggestions"]) == 3 for r in recs)
    ps = [{"start": 0, "text": "ab ba ab.", "context_ids": [DOC_ID] + VOCAB.encode("ba ab ")}]
    summ, rows = run_ksr(m2, vocab, ps, l_ctx=8)                       # l_ctx nhỏ -> đi qua nhánh cửa sổ trượt
    assert summ["chars"] == 9 and 0 <= summ["ksr"] < 1 and summ["keys"] <= 9
    out = write_eval(info["run_name"], "topk", score_topk(recs), tmp_path)
    write_eval(info["run_name"], "ksr", summ, tmp_path)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert set(data) == {"run_name", "topk", "ksr"}


@pytest.mark.parametrize("name", NAMES)
def test_reduced_precision_states(name):
    """Đánh giá dưới autocast (GPU: fp16 như eval_spec; CPU: bf16 thay thế): state sinh từng bước cùng dtype
    autocast — kernel causal_conv1d_update của mamba-ssm đòi conv_state cùng dtype đầu vào — và suggest chạy được,
    logits gần bản fp32."""
    from tests.test_models import DEVICE, _build
    model = _build(name).eval() if DEVICE.type == "cuda" else _tiny(name)          # GPU: cỡ thật, backend cuda
    dev, adt = (DEVICE, torch.float16) if DEVICE.type == "cuda" else (torch.device("cpu"), torch.bfloat16)
    vocab = VOCAB if DEVICE.type != "cuda" else CharVocab(ITOS + [chr(0x100 + i) for i in range(len(model.embedding.weight) - len(ITOS))])
    x = torch.tensor([vocab.encode("ab ba ab ba ab")], device=dev)
    with torch.no_grad():
        ref, _ = model.prefill(x)
        with torch.autocast(dev.type, dtype=adt):
            logits, state = model.prefill(x)
            logits, state = model.step(x[:, -1], state)
            empty = model.init_state(2, dev)
            words = suggest(model, vocab, "ab ba a", max_word_len=4)
    for t in _tensors(state) + _tensors(empty):
        assert t.dtype == adt
    assert len(words) == 3
    torch.testing.assert_close(model.prefill(x)[0], ref)            # ngoài autocast vẫn là fp32 như cũ
