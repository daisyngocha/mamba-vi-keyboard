"""Phân loại lỗi gợi ý đặc thù tiếng Việt (docs/eval_spec.md mục 6). Owner: DL (Cao Thu Hằng). Issue #25.

Trên các ca Top-1 SAI của typing_sim (đọc logs/<run>_suggest.jsonl do evaluate/ksr.py ghi), so gợi ý số 1 với target:
    1. bỏ 5 dấu thanh (U+0300 huyền, U+0301 sắc, U+0303 ngã, U+0309 hỏi, U+0323 nặng) mà giống  -> dau_thanh   đa / đã
    2. bỏ mọi dấu phụ (thêm ^, ˘, móc; đ -> d) mà giống                                         -> dau_chu     ân / ăn
    3. chỉ khác hoa/thường                                                                       -> viet_hoa    hà / Hà
    4. còn lại                                                                                   -> chu_goc     nắng / mưa
Thứ tự kiểm tra như trên (loại đầu tiên khớp được chọn).
Tách khỏi (1): cùng chữ, cùng dấu thanh, chỉ khác VỊ TRÍ đặt dấu (kiểu cũ/mới: "hóa" / "hoà", "thủy" / "thuỷ")
-> vi_tri_dau: cùng một từ, lỗi do quy ước chính tả lẫn lộn trong dữ liệu, không phải model chọn sai thanh. Hai trường hợp spec chưa nói, tách riêng để không lẫn vào
chu_goc: khác cả hoa/thường LẪN dấu (vd "Đã" / "đa") -> hoa_va_dau; không có gợi ý nào -> khong_goi_y.

Cách dùng:
    from evaluate.errors import classify
    classify("đa", "đã")   # 'dau_thanh'
    python -m evaluate.errors --run mamba_L256_s0 transformer_L256_s0     # ghi khóa "errors" vào logs/<run>_eval.json
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter
from pathlib import Path

TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}       # huyền, sắc, ngã, hỏi, nặng
CATEGORIES = ["vi_tri_dau", "dau_thanh", "dau_chu", "viet_hoa", "hoa_va_dau", "chu_goc", "khong_goi_y"]
LABELS = {"vi_tri_dau": "Khác vị trí dấu", "dau_thanh": "Sai dấu thanh", "dau_chu": "Sai dấu chữ", "viet_hoa": "Sai viết hoa",
          "hoa_va_dau": "Sai hoa + dấu", "chu_goc": "Sai chữ cái gốc", "khong_goi_y": "Không có gợi ý"}


def strip_tones(s: str) -> str:
    """Bỏ 5 dấu thanh, giữ dấu chữ (â, ă, ê, ô, ơ, ư, đ)."""
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", s) if c not in TONE_MARKS))


def tones(s: str) -> list[str]:
    """Các dấu thanh theo thứ tự xuất hiện."""
    return [c for c in unicodedata.normalize("NFD", s) if c in TONE_MARKS]


def strip_marks(s: str) -> str:
    """Bỏ mọi dấu phụ (thanh + chữ) và đ -> d: 'Đường' -> 'Duong'."""
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def classify(pred: str | None, target: str) -> str | None:
    """Loại lỗi của gợi ý số 1 so với target; None nếu đúng (khớp chính xác)."""
    if not pred:
        return "khong_goi_y"
    pred, target = unicodedata.normalize("NFC", pred), unicodedata.normalize("NFC", target)
    if pred == target:
        return None
    if strip_tones(pred) == strip_tones(target):
        return "vi_tri_dau" if tones(pred) == tones(target) else "dau_thanh"
    if strip_marks(pred) == strip_marks(target):
        return "dau_chu"
    if pred.lower() == target.lower():
        return "viet_hoa"
    if strip_marks(pred).lower() == strip_marks(target).lower():
        return "hoa_va_dau"
    return "chu_goc"


def summarize(records: list[dict], n_examples: int = 5) -> dict:
    """records: dòng của <run>_suggest.jsonl {mode, target, suggestions}. Tỷ lệ từng loại trên các ca Top-1 sai,
    theo mode a, b và gộp; kèm vài ví dụ mỗi loại (gợi ý → target)."""
    out = {}
    for name in ["a", "b", "all"]:
        rows = [r for r in records if name == "all" or r["mode"] == name]
        wrong = [(r, classify(r["suggestions"][0] if r["suggestions"] else None, r["target"])) for r in rows]
        wrong = [(r, c) for r, c in wrong if c is not None]
        cnt = Counter(c for _, c in wrong)
        n = len(wrong)
        ex = {c: [f"{r['suggestions'][0] if r['suggestions'] else '∅'} → {r['target']}"
                  for r, cc in wrong if cc == c][:n_examples] for c in CATEGORIES if cnt[c]}
        out[name] = {"n_positions": len(rows), "n_top1_wrong": n,
                     "counts": {c: cnt[c] for c in CATEGORIES},
                     "rates": {c: (cnt[c] / n if n else 0.0) for c in CATEGORIES},
                     # ca sai do dấu/hoa mà từ đúng vẫn nằm trong Top-3 (người dùng chạm gợi ý 2–3 là xong)
                     "diacritic_or_case_but_in_top3": sum(r["target"] in r["suggestions"] for r, c in wrong
                                                          if c in ("vi_tri_dau", "dau_thanh", "dau_chu", "viet_hoa",
                                                                   "hoa_va_dau")),
                     # trong nhóm chu_goc: target viết hoa (tên riêng / đầu câu) — khó đoán từ ngữ cảnh
                     "chu_goc_target_capitalized": sum(r["target"][:1].isupper() for r, c in wrong if c == "chu_goc"),
                     "examples": ex}
    return out


def load_suggest(run: str, log_dir: Path) -> list[dict]:
    p = Path(log_dir) / f"{run}_suggest.jsonl"
    if not p.exists():
        raise SystemExit(f"Không thấy {p} — chạy evaluate.ksr (eval_run) cho lượt {run} trước")
    return [json.loads(line) for line in open(p, encoding="utf-8")]


def main():
    from evaluate.common import write_eval
    ap = argparse.ArgumentParser(description="Phân loại lỗi Top-1 (eval_spec mục 6) từ logs/<run>_suggest.jsonl")
    ap.add_argument("--run", nargs="+", required=True)
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--no_write", action="store_true", help="chỉ in, không ghi vào _eval.json")
    args = ap.parse_args()
    for run in args.run:
        res = summarize(load_suggest(run, Path(args.log_dir)))
        if not args.no_write:
            write_eval(run, "errors", res, Path(args.log_dir))
        a = res["all"]
        print(f"[{run}] Top-1 sai {a['n_top1_wrong']}/{a['n_positions']}: " + " · ".join(
            f"{LABELS[c]} {a['counts'][c]} ({a['rates'][c]:.1%})" for c in CATEGORIES if a["counts"][c]))
        for c, ex in a["examples"].items():
            print(f"   {LABELS[c]:16s} {' | '.join(ex)}")


if __name__ == "__main__":
    main()
