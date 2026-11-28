"""evaluate/errors.py — phân loại lỗi tiếng Việt (eval_spec mục 6, issue #25)."""
import unicodedata

import pytest

from evaluate.errors import CATEGORIES, classify, strip_marks, strip_tones, summarize


@pytest.mark.parametrize("pred, target, want", [
    ("đa", "đã", "dau_thanh"),          # ví dụ trong issue #25
    ("ân", "ăn", "dau_chu"),
    ("hà", "Hà", "viet_hoa"),
    ("nắng", "mưa", "chu_goc"),
    ("người", "ngươi", "dau_thanh"),    # chỉ khác thanh huyền
    ("đi", "di", "dau_chu"),            # đ -> d
    ("ơ", "o", "dau_chu"),
    ("hóa", "hòa", "dau_thanh"),        # sắc vs huyền
    ("thủy", "thuỷ", "vi_tri_dau"),     # cùng thanh hỏi, khác vị trí đặt dấu
    ("Đã", "đa", "hoa_va_dau"),
    ("", "x", "khong_goi_y"),
    (None, "x", "khong_goi_y"),
    ("đã", "đã", None),                 # đúng
])
def test_classify(pred, target, want):
    assert classify(pred, target) == want


def test_nfd_input_same_as_nfc():
    nfd = unicodedata.normalize("NFD", "đã")
    assert classify(nfd, "đã") is None
    assert classify(unicodedata.normalize("NFD", "đa"), "đã") == "dau_thanh"


def test_strip():
    assert strip_tones("Đường hoà") == "Đương hoa"
    assert strip_marks("Đường hoà") == "Duong hoa"


def test_summarize():
    recs = [
        {"mode": "a", "target": "đã", "suggestions": ["đa", "đã", "đó"]},     # dấu thanh, đúng ở Top-3
        {"mode": "a", "target": "ăn", "suggestions": ["ăn", "ân", "anh"]},    # Top-1 đúng -> không tính
        {"mode": "b", "target": "Hà", "suggestions": ["hà", "Hà", "là"]},     # viết hoa
        {"mode": "b", "target": "Katz", "suggestions": ["The", "Mark", "A"]}, # chữ gốc, target viết hoa
        {"mode": "b", "target": "x", "suggestions": []},
    ]
    s = summarize(recs)
    assert s["all"]["n_positions"] == 5 and s["all"]["n_top1_wrong"] == 4
    assert s["a"]["counts"]["dau_thanh"] == 1 and s["a"]["n_top1_wrong"] == 1
    assert s["b"]["counts"] == {**{c: 0 for c in CATEGORIES}, "viet_hoa": 1, "chu_goc": 1, "khong_goi_y": 1}
    assert abs(sum(s["all"]["rates"].values()) - 1) < 1e-9
    assert s["all"]["diacritic_or_case_but_in_top3"] == 2
    assert s["all"]["chu_goc_target_capitalized"] == 1
    assert s["all"]["examples"]["dau_thanh"] == ["đa → đã"]
