"""Giao diện dữ liệu dùng chung. Owner: DL.

Hợp đồng — Hướng dẫn Tuần 1, mục 3.2 (mọi người code theo đúng giao diện này):

    data_dir/  (Kaggle Dataset `mamba-btl-data`, đọc trực tiếp từ /kaggle/input/…/mamba-btl-data — utils.kaggle_input)
        train.bin  val.bin  test.bin   mảng uint16 (vocab ≈ 200–300 ký tự > 255 nên không dùng uint8),
                                       các bài nối nhau, mỗi bài mở đầu bằng id 1 = <doc>
        sample.bin                     ≈ 1M ký tự đầu của train.bin, để thử code sớm
        vocab.json                     {"itos": ["<unk>", "<doc>", " ", "n", ...]}  id 0 = <unk>, id 1 = <doc>
        meta.json                      vocab_size, số ký tự mỗi tập, tỷ lệ <unk>, phiên bản dataset, seed
        typing_sim.jsonl               (tùy chọn) tập mô phỏng gõ phím lấy từ test

    data = build_data(cfg.data)
    data.vocab.encode(text) -> list[int]      # NFC, GIỮ CHỮ HOA (quyết định Q2, docs/decisions.md)
    data.vocab.decode(ids)  -> str
    x, y = data.get_batch(split, batch_size, seq_len, step, seed)
        # split ∈ {"train", "val", "test"}; x, y: torch.int64 (B, L) trên CPU; y = x dịch 1 ký tự.
        # Vị trí cắt cửa sổ CHỈ phụ thuộc (seed, step, split) -> resume không cần lưu trạng thái dataloader,
        # và gọi lại cùng (seed, step) luôn ra cùng batch.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import torch

from utils import kaggle_input

UNK, DOC = "<unk>", "<doc>"
UNK_ID, DOC_ID = 0, 1
KAGGLE_DATASET = "mamba-btl-data"
LOCAL_DATA_DIR = Path("data/processed")
SPLIT_SALT = {"train": 0, "val": 1, "test": 2}
_MAX_CODEPOINT = 0x110000


def normalize(text: str) -> str:
    """Chuẩn hóa Unicode NFC. KHÔNG chuyển chữ thường."""
    return unicodedata.normalize("NFC", text)


class CharVocab:
    def __init__(self, itos: list[str]):
        assert itos[UNK_ID] == UNK and itos[DOC_ID] == DOC, "id 0 phải là <unk>, id 1 phải là <doc>"
        assert all(len(c) == 1 for c in itos[2:]), "mỗi phần tử từ vựng (trừ <unk>, <doc>) là đúng 1 ký tự"
        assert len(itos) <= np.iinfo(np.uint16).max
        self.itos = itos
        self.stoi = {c: i for i, c in enumerate(itos)}
        self._lut = None  # bảng tra codepoint -> id, tạo khi cần (≈ 2,2 MB)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, UNK_ID) for c in normalize(text)]

    def encode_array(self, text: str, normalized: bool = False) -> np.ndarray:
        """Bản vectorized của encode (dùng cho văn bản dài), trả về np.uint16."""
        if self._lut is None:
            self._lut = np.full(_MAX_CODEPOINT, UNK_ID, dtype=np.uint16)
            for i, c in enumerate(self.itos[2:], start=2):
                self._lut[ord(c)] = i
        text = text if normalized else normalize(text)
        cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        return self._lut[cps]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    @classmethod
    def load(cls, path: str | Path) -> "CharVocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["itos"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")


def window_starts(n_tokens: int, batch_size: int, seq_len: int, step: int, seed: int, split: str) -> torch.Tensor:
    """B vị trí bắt đầu cửa sổ dài seq_len + 1, chỉ phụ thuộc (seed, step, split)."""
    if n_tokens < seq_len + 2:
        raise ValueError(f"Tập {split} chỉ có {n_tokens} ký tự, không cắt được cửa sổ L={seq_len}")
    g = torch.Generator().manual_seed(((seed * 1_000_003 + step) * 3 + SPLIT_SALT[split]) % (2**63 - 1))
    return torch.randint(n_tokens - seq_len - 1, (batch_size,), generator=g)


def resolve_data_dir(data_dir: str | None) -> Path:
    if data_dir:
        return Path(data_dir)
    return kaggle_input(KAGGLE_DATASET) or LOCAL_DATA_DIR


class WikiCharData:
    """Dữ liệu Wikipedia tiếng Việt đã xử lý bởi data/prepare.py. Đọc bằng np.memmap, không nạp hết vào RAM."""

    def __init__(self, data_dir: Path, train_file: str = "train.bin"):
        self.data_dir = Path(data_dir)
        self.vocab = CharVocab.load(self.data_dir / "vocab.json")
        files = {"train": train_file, "val": "val.bin", "test": "test.bin"}
        self.splits: dict[str, np.memmap] = {}
        for split, name in files.items():
            path = self.data_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Không thấy {path}")
            self.splits[split] = np.memmap(path, dtype=np.uint16, mode="r")
        meta_path = self.data_dir / "meta.json"
        self.meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        if self.meta and self.meta.get("vocab_size") != len(self.vocab):
            raise ValueError("meta.json và vocab.json lệch vocab_size — dataset bị trộn phiên bản")

    def n_tokens(self, split: str) -> int:
        return len(self.splits[split])

    def get_batch(self, split: str, batch_size: int, seq_len: int, step: int, seed: int):
        data = self.splits[split]
        ix = window_starts(len(data), batch_size, seq_len, step, seed, split).tolist()
        w = torch.from_numpy(np.stack([data[i : i + seq_len + 1] for i in ix]).astype(np.int64))
        return w[:, :-1].contiguous(), w[:, 1:].contiguous()


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
