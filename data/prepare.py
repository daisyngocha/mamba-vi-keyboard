"""Chuẩn bị dữ liệu Wikipedia tiếng Việt. Owner: DL.

Chạy từ thư mục gốc repo (Kaggle: notebook CPU, bật Internet; ≈ 15–30 phút):
    python -m data.prepare --out /kaggle/working/processed
    python -m data.prepare --out data/processed --train_chars 2_000_000 --val_chars 200_000 --test_chars 200_000  # chạy thử nhanh
Sau đó tạo phiên bản mới của Kaggle Dataset `mamba-btl-data` từ thư mục --out.

Đặc tả — Mô tả bài toán mục 2, Hướng dẫn Tuần 1 mục 5:
1. Tải HF `wikimedia/wikipedia` cấu hình `20231101.vi` (streaming, trộn bằng buffer theo seed).
2. Khảo sát ≈ 5.000 bài đầu luồng -> survey/: phân bố độ dài, tỷ lệ bài < 2.000 ký tự, tần suất ký tự.
3. Làm sạch từng bài: Unicode NFC, bỏ ký tự vô hình, gộp khoảng trắng và dòng trống thừa. GIỮ CHỮ HOA.
   Bỏ bài < 2.000 ký tự và bài trùng nội dung.
4. Chia theo BÀI VIẾT (xáo bằng seed): test ≈ 1M, val ≈ 1M, train ≈ 50M ký tự. Một bài chỉ thuộc một tập.
5. Từ vựng xây từ TRAIN: ký tự phổ biến nhất phủ ≥ 99,9% (≈ 200–300 ký tự); còn lại -> <unk>.
   id 0 = <unk>, id 1 = <doc>.
6. Mã hóa: mỗi bài mở đầu bằng <doc>, nối liền -> train/val/test.bin (uint16); sample.bin = 1M ký tự đầu train.
7. meta.json (vocab_size, số ký tự, tỷ lệ <unk>, phiên bản dataset, seed, tham số) và splits.json (id bài mỗi tập).
8. typing_sim.jsonl: ≈ 2.000 vị trí ngẫu nhiên trong test cho mô phỏng gõ phím — định dạng NHÁP, chốt với ĐG
   trong docs/eval_spec.md:
       {"id": 0, "mode": "a"|"b", "typed": "<văn bản đã gõ>", "prefix": "<phần từ đã gõ dở>", "target": "<từ đúng>",
        "doc": <id bài>}
   mode a (hoàn thành từ): typed kết thúc bằng prefix (≥ 1 ký tự); mode b (từ tiếp theo): prefix = "", typed kết thúc
   bằng khoảng trắng. "Từ" = âm tiết gồm toàn chữ cái, tách theo khoảng trắng.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # cho phép chạy `python data/prepare.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.loader import DOC, DOC_ID, UNK, UNK_ID, CharVocab, normalize  # noqa: E402

HF_DATASET, HF_CONFIG = "wikimedia/wikipedia", "20231101.vi"
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)
_SPACES = re.compile(r"[ \t  -   　]+")
_WORD = re.compile(r"(?<!\S)(\w+)")


# ---------------------------------------------------------------- làm sạch
def clean_article(text: str) -> str:
    """NFC, bỏ ký tự vô hình, gộp khoảng trắng, bỏ dòng trống. Giữ chữ hoa và xuống dòng giữa các đoạn."""
    text = normalize(text).translate(_INVISIBLE).replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def char_counts(texts, chunk_chars: int = 5_000_000) -> np.ndarray:
    counts = np.zeros(0x110000, dtype=np.int64)
    buf, n = [], 0

    def flush():
        if buf:
            cps = np.frombuffer("".join(buf).encode("utf-32-le"), dtype=np.uint32)
            counts[:] += np.bincount(cps, minlength=0x110000)
            buf.clear()

    for t in texts:
        buf.append(t)
        n += len(t)
        if n >= chunk_chars:
            flush()
            n = 0
    flush()
    return counts


def coverage_size(counts: np.ndarray, coverage: float) -> int:
    """Số ký tự phổ biến nhất cần để phủ `coverage` tổng số ký tự."""
    c = np.sort(counts[counts > 0])[::-1]
    return int(np.searchsorted(np.cumsum(c) / c.sum(), coverage) + 1)


# ---------------------------------------------------------------- từ vựng
def build_vocab(train_texts, coverage: float) -> tuple[CharVocab, dict]:
    counts = char_counts(train_texts)
    cps = np.nonzero(counts)[0]
    order = cps[np.lexsort((cps, -counts[cps]))]          # tần suất giảm dần, hòa thì theo codepoint
    cum = np.cumsum(counts[order]) / counts.sum()
    k = int(np.searchsorted(cum, coverage) + 1)
    kept = order[:k]
    itos = [UNK, DOC] + [chr(c) for c in kept]
    stats = {"distinct_chars_train": int(len(cps)), "kept_chars": k,
             "coverage_train": round(float(cum[k - 1]), 6),
             "dropped_examples": "".join(chr(c) for c in order[k:k + 40])}
    return CharVocab(itos), stats


# ---------------------------------------------------------------- tải + lọc
def stream_articles(args):
    from datasets import load_dataset
    ds = load_dataset(HF_DATASET, HF_CONFIG, split="train", streaming=True)
    return ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)


def collect(args, out: Path) -> tuple[list[dict], dict]:
    target = int((args.train_chars + args.val_chars + args.test_chars) * args.overshoot)
    kept, seen_hash = [], set()
    survey_lens, survey_texts = [], []
    n_seen = n_short = n_dup = total = 0
    t0 = time.time()
    for ex in stream_articles(args):
        n_seen += 1
        text = clean_article(ex["text"])
        if n_seen <= args.survey_n:
            survey_lens.append(len(text))
            survey_texts.append(text)
            if n_seen == args.survey_n:
                write_survey(out / "survey", survey_lens, survey_texts, args)
        if len(text) < args.min_chars:
            n_short += 1
            continue
        h = hashlib.sha1(text.encode("utf-8")).digest()
        if h in seen_hash:
            n_dup += 1
            continue
        seen_hash.add(h)
        kept.append({"id": ex.get("id"), "title": ex.get("title"), "text": text})
        total += len(text) + 1
        if n_seen % 20_000 == 0:
            print(f"  đã duyệt {n_seen:,} bài, giữ {len(kept):,} bài, {total/1e6:.1f}M ký tự ({time.time()-t0:.0f}s)")
        if total >= target:
            break
    if n_seen < args.survey_n:
        write_survey(out / "survey", survey_lens, survey_texts, args)
    stats = {"articles_scanned": n_seen, "articles_kept": len(kept), "dropped_short": n_short,
             "dropped_duplicate": n_dup, "chars_collected": total}
    print(f"Thu thập xong: {stats}")
    if total < args.train_chars + args.val_chars + args.test_chars:
        print("CẢNH BÁO: không đủ ký tự cho quy mô yêu cầu — tập train sẽ nhỏ hơn dự kiến.")
    return kept, stats


def write_survey(d: Path, lens: list[int], texts: list[str], args) -> None:
    d.mkdir(parents=True, exist_ok=True)
    lens_np = np.array(lens)
    counts = char_counts(texts)
    survey = {
        "n_articles": len(lens),
        "length_percentiles": {p: int(np.percentile(lens_np, p)) for p in (10, 25, 50, 75, 90, 99)},
        "mean_length": float(lens_np.mean()),
        f"share_below_{args.min_chars}": round(float((lens_np < args.min_chars).mean()), 4),
        f"chars_share_in_articles_below_{args.min_chars}": round(float(lens_np[lens_np < args.min_chars].sum()
                                                                        / max(1, lens_np.sum())), 4),
        "distinct_chars": int((counts > 0).sum()),
        "chars_for_coverage": {str(c): coverage_size(counts, c) for c in (0.99, 0.999, 0.9999)},
    }
    (d / "survey.json").write_text(json.dumps(survey, indent=2, ensure_ascii=False), encoding="utf-8")
    cps = np.nonzero(counts)[0]
    order = cps[np.argsort(-counts[cps], kind="stable")]
    total = counts.sum()
    with open(d / "char_freq.csv", "w", encoding="utf-8") as f:
        f.write("rank,char,codepoint,count,share,cum_share\n")
        cum = 0
        for r, c in enumerate(order, 1):
            cum += counts[c]
            f.write(f"{r},{json.dumps(chr(c), ensure_ascii=False)},U+{c:04X},{counts[c]},"
                    f"{counts[c]/total:.6f},{cum/total:.6f}\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.log10(np.maximum(lens_np, 1)), bins=60)
        ax.axvline(np.log10(args.min_chars), color="red", linestyle="--", label=f"{args.min_chars} ký tự")
        ax.set(xlabel="log10(độ dài bài, ký tự)", ylabel="số bài", title=f"Độ dài {len(lens):,} bài Wikipedia vi")
        ax.legend()
        fig.tight_layout()
        fig.savefig(d / "length_hist.png", dpi=150)
        plt.close(fig)
    except ImportError:
        pass
    print(f"Khảo sát {len(lens):,} bài -> {d}: {survey}")


# ---------------------------------------------------------------- chia tập
def split_articles(articles: list[dict], args) -> dict[str, list[dict]]:
    idx = list(range(len(articles)))
    random.Random(args.seed).shuffle(idx)
    quota = [("test", args.test_chars), ("val", args.val_chars), ("train", args.train_chars)]
    splits, q, filled = {k: [] for k, _ in quota}, 0, 0
    for i in idx:
        name, cap = quota[q]
        splits[name].append(articles[i])
        filled += len(articles[i]["text"]) + 1
        if filled >= cap:
            q, filled = q + 1, 0
            if q == len(quota):
                break
    return splits


# ---------------------------------------------------------------- mã hóa + ghi
def write_split(path: Path, docs: list[dict], vocab: CharVocab) -> dict:
    n_tok = n_unk = 0
    with open(path, "wb") as f:
        for d in docs:
            arr = np.concatenate([np.array([DOC_ID], dtype=np.uint16), vocab.encode_array(d["text"], normalized=True)])
            arr.tofile(f)
            n_tok += len(arr)
            n_unk += int((arr == UNK_ID).sum())
    return {"n_chars": n_tok, "n_articles": len(docs), "unk_rate": round(n_unk / max(1, n_tok), 6)}


def make_typing_sim(test_docs: list[dict], n: int, max_context: int, seed: int) -> list[dict]:
    rng = random.Random(seed + 1)
    weights = [len(d["text"]) for d in test_docs]
    items, tries = [], 0
    while len(items) < n and tries < n * 50:
        tries += 1
        di = rng.choices(range(len(test_docs)), weights=weights)[0]
        text = test_docs[di]["text"]
        m = _WORD.search(text, rng.randrange(len(text)))
        if not m or m.start() < 32 or not m.group(1).isalpha():
            continue
        target, start = m.group(1), m.start()
        mode = "a" if len(items) % 2 == 0 else "b"
        if mode == "a" and len(target) < 2:
            continue
        prefix = target[: rng.randint(1, len(target) - 1)] if mode == "a" else ""
        items.append({"id": len(items), "mode": mode, "typed": text[max(0, start - max_context):start] + prefix,
                      "prefix": prefix, "target": target, "doc": test_docs[di]["id"]})
    return items


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_chars", type=int, default=50_000_000)
    ap.add_argument("--val_chars", type=int, default=1_000_000)
    ap.add_argument("--test_chars", type=int, default=1_000_000)
    ap.add_argument("--sample_chars", type=int, default=1_000_000)
    ap.add_argument("--min_chars", type=int, default=2000)
    ap.add_argument("--coverage", type=float, default=0.999)
    ap.add_argument("--survey_n", type=int, default=5000)
    ap.add_argument("--shuffle_buffer", type=int, default=10_000)
    ap.add_argument("--overshoot", type=float, default=1.02, help="thu thêm ký tự để bù phần dư khi chia")
    ap.add_argument("--typing_sim_n", type=int, default=2000, help="0 = bỏ qua")
    ap.add_argument("--typing_sim_context", type=int, default=4096)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    articles, collect_stats = collect(args, out)
    splits = split_articles(articles, args)
    vocab, vocab_stats = build_vocab((d["text"] for d in splits["train"]), args.coverage)
    vocab.save(out / "vocab.json")
    print(f"Từ vựng: {len(vocab)} id ({vocab_stats})")
    if not 200 <= len(vocab) <= 300:
        print("LƯU Ý: vocab_size nằm ngoài khoảng 200–300 dự kiến — ghi vào ghi chú dữ liệu.")

    split_stats = {name: write_split(out / f"{name}.bin", splits[name], vocab) for name in ("train", "val", "test")}
    train = np.memmap(out / "train.bin", dtype=np.uint16, mode="r")
    np.asarray(train[: args.sample_chars]).tofile(out / "sample.bin")
    split_stats["sample"] = {"n_chars": int(min(len(train), args.sample_chars)), "source": "đầu train.bin"}
    print(f"Các tập: {split_stats}")

    ids = {k: [d["id"] for d in v] for k, v in splits.items()}
    assert not (set(ids["train"]) & set(ids["val"]) or set(ids["train"]) & set(ids["test"])
                or set(ids["val"]) & set(ids["test"])), "rò rỉ: một bài nằm ở nhiều tập"
    (out / "splits.json").write_text(json.dumps(ids), encoding="utf-8")

    if args.typing_sim_n:
        sim = make_typing_sim(splits["test"], args.typing_sim_n, args.typing_sim_context, args.seed)
        with open(out / "typing_sim.jsonl", "w", encoding="utf-8") as f:
            for it in sim:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"typing_sim.jsonl: {len(sim)} vị trí")

    try:
        import datasets
        ds_version = datasets.__version__
    except ImportError:
        ds_version = None
    meta = {
        "vocab_size": len(vocab), "splits": split_stats,
        "dataset": f"{HF_DATASET} {HF_CONFIG}", "license": "CC BY-SA", "datasets_lib": ds_version,
        "normalization": "NFC, giữ chữ hoa, gộp khoảng trắng, bỏ dòng trống", "min_article_chars": args.min_chars,
        "coverage": args.coverage, "vocab": vocab_stats, "collect": collect_stats, "seed": args.seed,
        "args": vars(args), "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prepare_time_s": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Xong sau {meta['prepare_time_s']}s -> {out}")


if __name__ == "__main__":
    main()
