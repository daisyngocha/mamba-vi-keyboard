"""Giao diện dữ liệu dùng chung. Owner: DL (Ngô Thị Thu Hằng).

Hợp đồng — Hướng dẫn Tuần 1, mục 3.2 (mọi người code theo đúng giao diện này):

    data_dir/  (Kaggle Dataset `mamba-btl-data`, đọc trực tiếp từ /kaggle/input/…/mamba-btl-data — utils.kaggle_input)
        train.bin  val.bin  test.bin   mảng uint16 (vocab ≈ 200–300 ký tự > 255 nên không dùng uint8),
                                       các bài nối nhau, mỗi bài mở đầu bằng id 1 = <doc>
        sample.bin                     ≈ 1M ký tự lấy từ train, để thử code sớm (N2)
        vocab.json                     {"itos": ["<unk>", "<doc>", " ", "a", "A", ...]}  id 0 = <unk>, id 1 = <doc>
        meta.json                      vocab_size, số ký tự mỗi tập, tỷ lệ <unk>, phiên bản dataset, seed

    data = build_data(cfg.data)
    data.vocab.encode(text) -> list[int]      # NFC, GIỮ CHỮ HOA (quyết định 19/11, docs/decisions.md)
    data.vocab.decode(ids)  -> str
    x, y = data.get_batch(split, batch_size, seq_len, step, seed)
        # split ∈ {"train", "val", "test"}; x, y: torch.int64 (B, L) trên CPU; y = x dịch 1 ký tự.
        # Vị trí cắt cửa sổ CHỈ phụ thuộc (seed, step, split) -> resume không cần lưu trạng thái dataloader,
        # và gọi lại cùng (seed, step) luôn ra cùng batch.

Tự kiểm tra trước khi mở PR (đưa vào tests/test_loader.py):
    - decode(encode(text)) == text với text chỉ gồm ký tự trong vocab
    - tỷ lệ <unk> trên val < 0,1%
    - cùng seed + step -> cùng batch; khác step -> khác batch
    - y[:, :-1] == x[:, 1:]
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import torch

from utils import kaggle_input

UNK, DOC = "<unk>", "<doc>"
UNK_ID, DOC_ID = 0, 1
KAGGLE_DATASET = "mamba-btl-data"
LOCAL_DATA_DIR = Path("data/processed")


def normalize(text: str) -> str:
    """Chuẩn hóa Unicode NFC. KHÔNG chuyển chữ thường."""
    return unicodedata.normalize("NFC", text)


class CharVocab:
    def __init__(self, itos: list[str]):
        assert itos[UNK_ID] == UNK and itos[DOC_ID] == DOC, "id 0 phải là <unk>, id 1 phải là <doc>"
        self.itos = itos
        self.stoi = {c: i for i, c in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, UNK_ID) for c in normalize(text)]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["itos"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")


def resolve_data_dir(data_dir: str | None) -> Path:
    if data_dir:
        return Path(data_dir)
    return kaggle_input(KAGGLE_DATASET) or LOCAL_DATA_DIR


class WikiCharData:
    """Dữ liệu Wikipedia tiếng Việt đã xử lý bởi data/prepare.py."""

    def __init__(self, data_dir: Path, train_file: str = "train.bin"):
        self.data_dir = data_dir
        self.vocab = CharVocab.load(data_dir / "vocab.json")
        # TODO(DL): mở 3 tập bằng np.memmap(..., dtype=np.uint16, mode="r") — không đọc hết vào RAM.
        #   self.splits = {"train": <memmap train_file>, "val": <memmap val.bin>, "test": <memmap test.bin>}
        raise NotImplementedError("TODO(DL): WikiCharData — mở .bin bằng np.memmap")

    def get_batch(self, split: str, batch_size: int, seq_len: int, step: int, seed: int):
        # TODO(DL): sinh B vị trí bắt đầu bằng torch.Generator().manual_seed(f(seed, step, split)),
        #   cắt cửa sổ dài seq_len + 1 từ memmap, chuyển int64 -> x = w[:, :-1], y = w[:, 1:].
        raise NotImplementedError("TODO(DL): get_batch theo mục 3.2")


def build_data(cfg_data):
    if cfg_data.source == "synthetic":
        from data.synthetic import SyntheticCharData
        return SyntheticCharData()
    if cfg_data.source == "wiki_vi":
        root = resolve_data_dir(cfg_data.data_dir)
        if not (root / "vocab.json").exists():
            raise FileNotFoundError(f"Không thấy {root}/vocab.json — Add Input Kaggle Dataset mamba-btl-data "
                                    "hoặc chạy data/prepare.py.")
        return WikiCharData(root, cfg_data.train_file)
    raise ValueError(f"Unknown data source: {cfg_data.source}")
