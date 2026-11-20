"""Kiểm tra data/prepare.py + WikiCharData trên luồng bài giả (không cần mạng)."""
import json
import random
import sys

import numpy as np
import pytest
import torch

import data.prepare as prep
from data.loader import DOC_ID, UNK_ID, WikiCharData

SYLL = ["Hôm", "nay", "trời", "nắng", "đẹp", "Hà", "Nội", "Việt", "Nam", "người", "được", "những", "của", "và",
        "là", "có", "trong", "một", "các", "năm", "Ở", "Ổn", "định", "đã", "đang", "mưa", "phố", "1945", "km²"]


def fake_stream(n=400, seed=0):
    rng = random.Random(seed)
    for i in range(n):
        n_words = rng.choice([50, 400, 900]) if i != 10 else 900  # có bài < 2.000 ký tự để kiểm tra lọc
        words = " ".join(rng.choice(SYLL) for _ in range(n_words))
        text = words.replace(" và ", " và ").replace(" của ", "\r\n\r\n của  ") + " ​"
        if i % 37 == 0:
            text += " \U0001F600 ⅞"  # ký tự hiếm -> <unk>
        yield {"id": str(i), "title": f"Bài {i}", "text": text}
        if i == 10:
            yield {"id": "dup", "title": "trùng", "text": text + "  "}  # trùng sau khi làm sạch


def test_clean_article():
    s = "Hôm\u00a0nay  trời\r\n\r\n\r\n  đe\u0323p\u200b "  # NFD "ẹ", NBSP, zero-width space
    assert prep.clean_article(s) == "Hôm nay trời\nđẹp"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    out = tmp_path_factory.mktemp("processed")
    prep.stream_articles = lambda args: fake_stream()
    argv = ["prepare", "--out", str(out), "--train_chars", "300000", "--val_chars", "30000", "--test_chars",
            "30000", "--sample_chars", "50000", "--survey_n", "100", "--typing_sim_n", "200"]
    old, sys.argv = sys.argv, argv
    try:
        prep.main()
    finally:
        sys.argv = old
    return out


def test_outputs_and_meta(prepared):
    for f in ["train.bin", "val.bin", "test.bin", "sample.bin", "vocab.json", "meta.json", "splits.json",
              "typing_sim.jsonl", "survey/survey.json", "survey/char_freq.csv"]:
        assert (prepared / f).exists(), f
    meta = json.loads((prepared / "meta.json").read_text())
    assert meta["collect"]["dropped_short"] > 0 and meta["collect"]["dropped_duplicate"] == 1
    assert meta["splits"]["val"]["unk_rate"] < 0.001
    ids = json.loads((prepared / "splits.json").read_text())
    assert not set(ids["train"]) & set(ids["test"])


def test_loader_on_prepared(prepared):
    data = WikiCharData(prepared)
    v = data.vocab
    assert "H" in v.stoi and "h" in v.stoi and "Ở" in v.stoi  # giữ chữ hoa
    text = "Hôm nay trời nắng đẹp Ở Hà Nội năm 1945"
    assert v.decode(v.encode(text)) == text
    assert np.array_equal(v.encode_array(text), np.array(v.encode(text), dtype=np.uint16))
    train = np.memmap(prepared / "train.bin", dtype=np.uint16, mode="r")
    assert train[0] == DOC_ID and (train == UNK_ID).mean() < 0.01
    x, y = data.get_batch("train", 8, 128, step=3, seed=0)
    assert x.shape == (8, 128) and x.dtype == torch.int64 and torch.equal(x[:, 1:], y[:, :-1])
    x2, _ = data.get_batch("train", 8, 128, step=3, seed=0)
    x3, _ = data.get_batch("train", 8, 128, step=4, seed=0)
    assert torch.equal(x, x2) and not torch.equal(x, x3)
    sample = WikiCharData(prepared, train_file="sample.bin")
    assert sample.n_tokens("train") == 50000


def test_typing_sim(prepared):
    rows = [json.loads(l) for l in (prepared / "typing_sim.jsonl").read_text().splitlines()]
    assert len(rows) == 200 and {r["mode"] for r in rows} == {"a", "b"}
    for r in rows:
        assert r["target"].isalpha() and r["typed"].endswith(r["prefix"])
        if r["mode"] == "b":
            assert r["prefix"] == "" and r["typed"][-1].isspace()
        else:
            assert 1 <= len(r["prefix"]) < len(r["target"]) and r["target"].startswith(r["prefix"])
