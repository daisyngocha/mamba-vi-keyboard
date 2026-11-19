"""Kiểm tra giao diện dữ liệu (mục 3.2). DL bổ sung test cho dữ liệu thật (decode/encode, tỷ lệ <unk>, trùng bài)."""
import torch

from data.loader import DOC_ID, UNK_ID, CharVocab
from data.synthetic import SyntheticCharData


def test_get_batch_contract():
    data = SyntheticCharData()
    x, y = data.get_batch("train", 4, 32, step=7, seed=0)
    assert x.shape == y.shape == (4, 32) and x.dtype == torch.int64
    assert torch.equal(x[:, 1:], y[:, :-1])
    x2, _ = data.get_batch("train", 4, 32, step=7, seed=0)
    x3, _ = data.get_batch("train", 4, 32, step=8, seed=0)
    assert torch.equal(x, x2) and not torch.equal(x, x3)


def test_vocab_keeps_case_and_nfc():
    v = CharVocab(["<unk>", "<doc>", "a", "A", "ắ"])
    assert v.stoi["<unk>"] == UNK_ID and v.stoi["<doc>"] == DOC_ID
    assert v.decode(v.encode("Aa")) == "Aa"
    assert v.encode("ắ") == [4]  # NFD "ắ" -> NFC


def test_kaggle_input_layouts(tmp_path, monkeypatch):
    """Kaggle gắn dataset ở /kaggle/input/<slug> hoặc /kaggle/input/datasets/<owner>/<slug>."""
    import utils
    monkeypatch.setattr(utils, "KAGGLE_INPUT", tmp_path)
    assert utils.kaggle_input("mamba-btl-data") is None
    new = tmp_path / "datasets" / "ngochadaisy" / "mamba-btl-data"
    new.mkdir(parents=True)
    assert utils.kaggle_input("mamba-btl-data") == new
    old = tmp_path / "mamba-btl-data"
    old.mkdir()
    assert utils.kaggle_input("mamba-btl-data") == old  # kiểu cũ được ưu tiên
