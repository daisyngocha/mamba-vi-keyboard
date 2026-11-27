"""Top-1 / Top-3 trên typing_sim.jsonl và KSR (tỷ lệ tiết kiệm phím gõ). Owner: ĐG (Ngô Thị Thanh Hằng). Issue #24.

Đúng theo docs/eval_spec.md mục 3–4:
- Top-k: với mỗi dòng {mode, typed, prefix, target} của typing_sim.jsonl, lấy 3 gợi ý cho `typed` (cắt còn l_ctx
  ký tự cuối). Đúng khi target khớp CHÍNH XÁC (cả hoa/thường) với gợi ý 1 (Top-1) / một trong 3 (Top-3).
  Báo riêng mode a, b và gộp; chỉ số phụ Top-3 không phân biệt hoa/thường.
- KSR: 50 đoạn × ≈ 400 ký tự trong test.bin (seed 0, bắt đầu ở đầu từ, cắt ở ranh giới từ, không cắt ngang bài).
  Gõ lại từng ký tự; trước mỗi phím trong một từ (kể cả chữ cái đầu) lấy 3 gợi ý; từ đúng có trong đó thì
  chạm chọn = 1 phím, chèn phần còn lại của từ, KHÔNG tự thêm khoảng trắng. Ký tự không phải chữ cái gõ tay.
  KSR = (tổng ký tự − tổng phím thực bấm) / tổng ký tự.
  Ngữ cảnh trước mỗi đoạn = phần bài viết đứng trước nó trong test.bin (như người dùng đã gõ trước đó), cắt còn
  l_ctx ký tự cuối cùng với phần đã gõ của đoạn.
- Kết quả ghi vào khóa "topk", "ksr" của logs/<run_name>_eval.json (evaluate.common.write_eval);
  chi tiết từng vị trí Top-k ghi logs/<run_name>_suggest.jsonl (DL dùng cho phân tích lỗi, eval_spec mục 6).

Chạy (thư mục gốc repo; trên Kaggle: Add Input output notebook có checkpoints/<run_name>/ckpt_latest.pt):
    python -m evaluate.ksr --run mamba_L256_s0                    # đủ Top-k (2.000 vị trí) + KSR (50 đoạn)
    python -m evaluate.ksr --run mamba_L256_s0 --limit 200 --skip_ksr          # chạy nhanh để thử
    python -m evaluate.ksr --ckpt path/to/ckpt_latest.pt --l_ctx 2048          # thí nghiệm ngoại suy (mục 6)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from data.loader import DOC_ID, UNK_ID, CharVocab, resolve_data_dir
from evaluate.common import TypingContext, autocast_ctx, find_checkpoint, is_letter, load_run, write_eval
from evaluate.suggest import split_prefix, suggest_from_state

SUGGEST_CFG = {"k": 3, "beam": 5, "max_word_len": 10}      # eval_spec mục 0 — cố định cho mọi model


# ---------------------------------------------------------------- Top-k
def score_topk(records: list[dict]) -> dict:
    """records: {mode, target, suggestions} -> Top-1, Top-3, Top-3 không phân biệt hoa/thường theo mode a, b, gộp."""
    out = {}
    for name, rows in [("a", [r for r in records if r["mode"] == "a"]),
                       ("b", [r for r in records if r["mode"] == "b"]), ("all", records)]:
        n = len(rows)
        if n == 0:
            continue
        top1 = sum(bool(r["suggestions"]) and r["suggestions"][0] == r["target"] for r in rows)
        top3 = sum(r["target"] in r["suggestions"] for r in rows)
        top3_nc = sum(r["target"].lower() in [s.lower() for s in r["suggestions"]] for r in rows)
        out[name] = {"n": n, "top1": top1 / n, "top3": top3 / n, "top3_nocase": top3_nc / n}
    return out


def run_topk(model, vocab, items: list[dict], l_ctx: int, dtype: str = "fp32", log_every: int = 200):
    device = next(model.parameters()).device
    records, t0 = [], time.perf_counter()
    for j, it in enumerate(items, 1):
        typed = it["typed"]
        prefix = split_prefix(typed)
        assert prefix == it["prefix"], f"typing_sim id {it.get('id')}: prefix trong file khác prefix tách từ typed"
        with autocast_ctx(device, dtype):
            ctx = TypingContext(model, vocab, l_ctx, typed[-l_ctx:])
            logits, state = ctx.current()
            sug = suggest_from_state(model, vocab, logits, state, prefix, **SUGGEST_CFG)
        words = [w for w, _ in sug]
        records.append({"id": it.get("id"), "mode": it["mode"], "prefix": prefix, "target": it["target"],
                        "suggestions": words, "hit1": bool(words) and words[0] == it["target"],
                        "hit3": it["target"] in words})
        if log_every and j % log_every == 0:
            s = score_topk(records)["all"]
            print(f"  top-k {j}/{len(items)} | top1 {s['top1']:.3f} top3 {s['top3']:.3f} "
                  f"| {1000 * (time.perf_counter() - t0) / j:.1f} ms/vị trí")
    return records, time.perf_counter() - t0


# ---------------------------------------------------------------- KSR
def simulate_typing(passage: str, suggest_fn) -> dict:
    """Mô phỏng gõ lại `passage`. suggest_fn(đã_gõ_trong_đoạn) -> list[str] gợi ý.
    Trả {chars, keys, selections}. Hàm thuần — test được không cần model."""
    n, i, keys, selections = len(passage), 0, 0, 0
    while i < n:
        if not is_letter(passage[i]):
            keys, i = keys + 1, i + 1                    # ký tự không phải chữ cái luôn gõ tay
            continue
        end = i
        while end < n and is_letter(passage[end]):
            end += 1
        word = passage[i:end]                            # i đang ở đầu từ
        while i < end:
            if word in suggest_fn(passage[:i]):          # trước mỗi phím trong từ, kể cả chữ cái đầu
                keys, selections, i = keys + 1, selections + 1, end   # chạm chọn = 1 phím, không thêm khoảng trắng
                break
            keys, i = keys + 1, i + 1
    return {"chars": n, "keys": keys, "selections": selections}


def ksr_passages(test_ids: np.ndarray, vocab: CharVocab, n: int = 50, length: int = 400, seed: int = 0,
                 max_context: int = 4096) -> list[dict]:
    """n đoạn ≈ length ký tự trong test.bin: bắt đầu ở đầu từ, kết thúc ở ranh giới từ, không chứa <doc>/<unk>,
    không chồng nhau. Kèm ngữ cảnh = tối đa max_context id đứng trước trong cùng bài (gồm <doc> đầu bài)."""
    rng = random.Random(seed)
    itos, N = vocab.itos, len(test_ids)
    out, used, tries = [], [], 0
    while len(out) < n and tries < n * 200:
        tries += 1
        s = rng.randrange(1, N - length - 1)
        while s < N - length and not (is_letter(itos[test_ids[s]]) and not is_letter(itos[test_ids[s - 1]])):
            s += 1                                        # tiến tới đầu từ gần nhất
        seg = test_ids[s:s + length].tolist()
        if len(seg) < length or DOC_ID in seg or UNK_ID in seg:
            continue
        e = len(seg)
        if is_letter(itos[seg[-1]]) and s + e < N and is_letter(itos[test_ids[s + e]]):
            while e > 0 and is_letter(itos[seg[e - 1]]):
                e -= 1                                    # từ cuối bị cắt ngang -> bỏ cả từ đó
        seg = seg[:e]
        if len(seg) < length // 2 or any(s < u_e and u_s < s + len(seg) for u_s, u_e in used):
            continue
        lo = max(0, s - max_context)
        ctx = test_ids[lo:s].tolist()
        if DOC_ID in ctx:                                 # chỉ lấy ngữ cảnh trong cùng bài, giữ <doc> đầu bài
            ctx = ctx[len(ctx) - 1 - ctx[::-1].index(DOC_ID):]
        used.append((s, s + len(seg)))
        out.append({"start": int(s), "text": vocab.decode(seg), "context_ids": ctx})
    return out


class ModelTypist:
    """suggest_fn cho simulate_typing: giữ TypingContext, mỗi lần gọi chỉ nạp thêm phần mới gõ."""

    def __init__(self, model, vocab, context_ids: list[int], l_ctx: int, dtype: str = "fp32"):
        self.model, self.vocab, self.dtype = model, vocab, dtype
        self.device = next(model.parameters()).device
        self.ctx = TypingContext(model, vocab, l_ctx, ids=context_ids[-l_ctx:])
        self.typed, self.calls = "", 0

    def __call__(self, typed: str) -> list[str]:
        assert typed.startswith(self.typed), "simulate_typing chỉ gõ thêm, không xóa"
        with autocast_ctx(self.device, self.dtype):
            self.ctx.push(typed[len(self.typed):])
            self.typed = typed
            logits, state = self.ctx.current()
            sug = suggest_from_state(self.model, self.vocab, logits, state, split_prefix(typed), **SUGGEST_CFG)
        self.calls += 1
        return [w for w, _ in sug]


def run_ksr(model, vocab, passages: list[dict], l_ctx: int, dtype: str = "fp32"):
    rows, t0, calls = [], time.perf_counter(), 0
    for j, p in enumerate(passages, 1):
        typist = ModelTypist(model, vocab, p["context_ids"], l_ctx, dtype)
        r = simulate_typing(p["text"], typist)
        calls += typist.calls
        rows.append({"start": p["start"], **r, "ksr": (r["chars"] - r["keys"]) / r["chars"]})
        if j % 10 == 0:
            c, k = sum(x["chars"] for x in rows), sum(x["keys"] for x in rows)
            print(f"  KSR {j}/{len(passages)} đoạn | KSR {(c - k) / c:.3f} | {time.perf_counter() - t0:.0f} s")
    chars, keys = sum(r["chars"] for r in rows), sum(r["keys"] for r in rows)
    summary = {"passages": len(rows), "chars": chars, "keys": keys, "selections": sum(r["selections"] for r in rows),
               "ksr": (chars - keys) / chars, "ksr_passage_mean": float(np.mean([r["ksr"] for r in rows])),
               "ksr_passage_std": float(np.std([r["ksr"] for r in rows])), "suggest_calls": calls,
               "time_s": round(time.perf_counter() - t0, 1)}
    return summary, rows


# ---------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Top-1/Top-3 (typing_sim) và KSR cho một checkpoint")
    ap.add_argument("--run", help="tên lượt, vd mamba_L1024_s0 (tự tìm checkpoint)")
    ap.add_argument("--ckpt", help="đường dẫn ckpt_latest.pt (thay cho --run)")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--data_dir", default=None, help="mặc định: Dataset mamba-btl-data trên Kaggle, rồi data/processed")
    ap.add_argument("--l_ctx", type=int, default=None, help="mặc định = L lúc huấn luyện (eval_spec mục 0)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N dòng đầu typing_sim (chạy thử)")
    ap.add_argument("--ksr_passages", type=int, default=50)
    ap.add_argument("--ksr_len", type=int, default=400)
    ap.add_argument("--skip_topk", action="store_true")
    ap.add_argument("--skip_ksr", action="store_true")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                    help="fp16 = autocast trên GPU như lúc huấn luyện (eval_spec mục 0, mặc định); CPU luôn fp32")
    ap.add_argument("--log_dir", default="logs")
    args = ap.parse_args()

    ckpt = Path(args.ckpt) if args.ckpt else find_checkpoint(args.run, args.checkpoint_dir)
    if ckpt is None or not ckpt.exists():
        raise SystemExit(f"Không thấy checkpoint của {args.run!r}. Trên Kaggle: Add Input output notebook của lượt đó.")
    model, vocab, info = load_run(ckpt)
    run_name = args.run or info["run_name"]
    l_ctx = args.l_ctx or info["seq_len"]
    data_dir = resolve_data_dir(args.data_dir)
    data_vocab = CharVocab.load(data_dir / "vocab.json")
    assert data_vocab.itos == vocab.itos, "vocab của checkpoint khác vocab.json của dữ liệu — sai phiên bản Dataset"
    suffix = "" if l_ctx == info["seq_len"] else f"_ctx{l_ctx}"
    device = next(model.parameters()).device
    print(f"[{run_name}] {info['model']} step {info['step']} | backend {info['backend']} | l_ctx {l_ctx} | "
          f"{device} {args.dtype} | data {data_dir}")
    common = {"checkpoint_step": info["step"], "l_ctx": l_ctx, "suggest_cfg": SUGGEST_CFG, "dtype": args.dtype,
              "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"}

    if not args.skip_topk:
        items = [json.loads(line) for line in open(data_dir / "typing_sim.jsonl", encoding="utf-8")]
        items = items[: args.limit] if args.limit else items
        records, secs = run_topk(model, vocab, items, l_ctx, args.dtype)
        res = {**common, **score_topk(records), "ms_per_item": round(1000 * secs / len(records), 1),
               "limit": args.limit}
        write_eval(run_name, "topk" + suffix, res, Path(args.log_dir))
        detail = Path(args.log_dir) / f"{run_name}_suggest{suffix}.jsonl"
        detail.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        print(f"Top-k: " + " · ".join(f"{m}: top1 {v['top1']:.3f} top3 {v['top3']:.3f}"
                                      for m, v in score_topk(records).items()) + f" -> {detail}")

    if not args.skip_ksr:
        test_ids = np.memmap(data_dir / "test.bin", dtype=np.uint16, mode="r")
        passages = ksr_passages(test_ids, vocab, args.ksr_passages, args.ksr_len, seed=0)
        summary, _ = run_ksr(model, vocab, passages, l_ctx, args.dtype)
        write_eval(run_name, "ksr" + suffix, {**common, **summary}, Path(args.log_dir))
        print(f"KSR {summary['ksr']:.3f} ({summary['chars']} ký tự, {summary['keys']} phím, "
              f"{summary['selections']} lần chạm chọn, {summary['time_s']} s)")


if __name__ == "__main__":
    main()
