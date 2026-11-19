"""Dữ liệu char-level. Owner: Ngô Thị Thu Hằng (lead đã viết phần đọc + nguồn synthetic).

CONTRACT với data/prepare_data.py:
    data/processed/
        vocab.json            {"itos": ["<unk>", " ", "a", ...]}   (id 0 = <unk>)
        train.bin / val.bin / test.bin   mảng uint16 các id ký tự, nối liền các bài viết,
                                         giữa 2 bài viết chèn "\n\n"
        stats.json            số ký tự mỗi split, kích thước vocab, tỉ lệ <unk>

get_batch(split) lấy ngẫu nhiên các cửa sổ dài seq_len+1 (kiểu nanoGPT) -> (x, y) dịch 1 ký tự.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import numpy as np
import torch


class CharVocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {c: i for i, c in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @staticmethod
    def normalize(text: str, lowercase: bool = True) -> str:
        text = unicodedata.normalize("NFC", text)
        return text.lower() if lowercase else text

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    @classmethod
    def load(cls, path: Path) -> "CharVocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8"))["itos"])

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos}, ensure_ascii=False), encoding="utf-8")


# Văn bản mẫu cho smoke test (không dùng để đánh giá).
_SYNTHETIC_TEXT = (
    "hôm nay trời rất nắng, chúng tôi đi học ở trường đại học bách khoa hà nội. "
    "mô hình mamba xử lý chuỗi với độ phức tạp tuyến tính theo độ dài. "
    "transformer dùng cơ chế attention nên chi phí tăng theo bình phương độ dài chuỗi. "
)


class CharDataset:
    def __init__(self, splits: dict[str, torch.Tensor], vocab: CharVocab, seq_len: int, batch_size: int):
        self.splits, self.vocab = splits, vocab
        self.seq_len, self.batch_size = seq_len, batch_size

    def get_batch(self, split: str, device: torch.device):
        data = self.splits[split]
        ix = torch.randint(len(data) - self.seq_len - 1, (self.batch_size,))
        x = torch.stack([data[i : i + self.seq_len] for i in ix]).long()
        y = torch.stack([data[i + 1 : i + 1 + self.seq_len] for i in ix]).long()
        if device.type == "cuda":
            return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        return x.to(device), y.to(device)


def build_dataset(cfg_data) -> CharDataset:
    if cfg_data.source == "synthetic":
        text = CharVocab.normalize(_SYNTHETIC_TEXT * 200, cfg_data.lowercase)
        vocab = CharVocab(["<unk>"] + sorted(set(text)))
        ids = torch.tensor(vocab.encode(text), dtype=torch.int64)
        cut = int(0.9 * len(ids))
        splits = {"train": ids[:cut], "val": ids[cut:], "test": ids[cut:]}
    elif cfg_data.source == "wiki_vi":
        root = Path(cfg_data.data_dir)
        if not (root / "vocab.json").exists():
            raise FileNotFoundError(f"Chưa có {root}/vocab.json — chạy data/prepare_data.py trước.")
        vocab = CharVocab.load(root / "vocab.json")
        splits = {
            s: torch.from_numpy(np.fromfile(root / f"{s}.bin", dtype=np.uint16).astype(np.int64))
            for s in ("train", "val", "test")
        }
    else:
        raise ValueError(f"Unknown data source: {cfg_data.source}")
    return CharDataset(splits, vocab, cfg_data.seq_len, cfg_data.batch_size)
