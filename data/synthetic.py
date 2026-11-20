"""Nguồn dữ liệu giả cho smoke test (TN). Cùng giao diện với data/loader.py; không dùng để đánh giá."""
from __future__ import annotations

import torch

from data.loader import DOC, DOC_ID, UNK, CharVocab, normalize, window_starts

_TEXT = (
    "Hôm nay trời rất nắng, chúng tôi đi học ở Trường Đại học Bách khoa Hà Nội. "
    "Mô hình Mamba xử lý chuỗi với độ phức tạp tuyến tính theo độ dài. "
    "Transformer dùng cơ chế attention nên chi phí tăng theo bình phương độ dài chuỗi. "
)


class SyntheticCharData:
    def __init__(self, repeats: int = 200):
        text = normalize(_TEXT)
        self.vocab = CharVocab([UNK, DOC] + sorted(set(text)))
        ids = torch.tensor([DOC_ID] + self.vocab.encode(text * repeats), dtype=torch.int64)
        cut = int(0.9 * len(ids))
        self.splits = {"train": ids[:cut], "val": ids[cut:], "test": ids[cut:]}

    def get_batch(self, split: str, batch_size: int, seq_len: int, step: int, seed: int):
        data = self.splits[split]
        ix = window_starts(len(data), batch_size, seq_len, step, seed, split)
        w = torch.stack([data[i : i + seq_len + 1] for i in ix.tolist()])
        return w[:, :-1].contiguous(), w[:, 1:].contiguous()
