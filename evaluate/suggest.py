"""Hàm gợi ý top-3 từ cho bàn phím. Owner: ĐG (Ngô Thị Thanh Hằng). Tuần 2, issue #24.

Đúng theo docs/eval_spec.md mục 0 và 2:
- "Từ" = các chữ cái liên tiếp (str.isalpha). Ký tự kết thúc từ = mọi ký tự không phải chữ cái.
- (a) hoàn thành từ: văn bản kết thúc giữa từ -> prefix = các chữ cái cuối; ứng viên = prefix + phần sinh thêm
      (phần sinh thêm được rỗng nếu prefix đã là một từ).       "Hôm nay trời đ"  -> đẹp · đang · đã
  (b) từ tiếp theo: văn bản kết thúc bằng ký tự kết thúc -> prefix = ""; ứng viên có ≥ 1 chữ cái.
                                                                 "Hôm nay trời "   -> mưa · nắng · đẹp
- Điểm(w) = Σ log p(ký tự sinh thêm) + log p(kết thúc | văn bản + w), p(kết thúc) = tổng xác suất các ký tự
  không phải chữ cái -> không ưu tiên từ cụt.
- Beam search theo ký tự, chỉ mở rộng bằng chữ cái, giữ `beam` nhánh, tối đa `max_word_len` ký tự sinh thêm.
  Trả `k` ứng viên khác nhau điểm cao nhất; hòa điểm -> thứ tự từ điển.
- Dừng sớm CHÍNH XÁC: mọi log-prob ≤ 0 nên điểm của một từ chỉ giảm khi sinh thêm; khi ứng viên thứ k đã cao hơn
  nhánh tốt nhất còn mở thì không nhánh nào vượt được nữa -> dừng, kết quả không đổi.

Cách dùng:
    from evaluate.suggest import suggest
    suggest(model, vocab, "Hôm nay trời đ")                     # ['đẹp', 'đang', 'đã'] (với model đã huấn luyện)
    suggest(model, vocab, typed, l_ctx=1024, return_scores=True) # [('đẹp', -1.23), ...]
Model nạp từ checkpoint: evaluate.common.load_run(path). Ngữ cảnh tăng dần khi gõ (KSR): evaluate.common.TypingContext
+ suggest_from_state() để không phải prefill lại cả văn bản sau mỗi phím.
"""
from __future__ import annotations

import torch

from data.loader import normalize
from evaluate.common import TypingContext, is_letter

_MASK_CACHE: dict = {}


def split_prefix(text: str) -> str:
    """Các chữ cái liên tiếp ở cuối văn bản (rỗng nếu văn bản kết thúc bằng ký tự kết thúc từ)."""
    i = len(text)
    while i > 0 and is_letter(text[i - 1]):
        i -= 1
    return text[i:]


def _letter_ids(vocab, device):
    """(id các chữ cái, mask chữ cái (V,)) theo vocab — tính một lần mỗi vocab/device."""
    key = (id(vocab), str(device))
    if key not in _MASK_CACHE:
        mask = torch.tensor([is_letter(c) for c in vocab.itos], device=device)
        _MASK_CACHE[key] = (mask.nonzero().squeeze(1), mask)
    return _MASK_CACHE[key]


@torch.no_grad()
def suggest_from_state(model, vocab, logits, state, prefix: str, k: int = 3, beam: int = 5,
                       max_word_len: int = 10) -> list[tuple[str, float]]:
    """Beam search từ (logits vị trí cuối (1, V), state) của văn bản đã gõ (đã gồm prefix).
    Trả [(từ, điểm)] sắp theo điểm giảm dần, tối đa k phần tử. Không sửa state truyền vào."""
    letters, is_let = _letter_ids(vocab, logits.device)
    n_let = len(letters)
    cands: dict[str, float] = {}

    def add_candidates(words, logp_rows, base_scores):
        term = torch.logsumexp(logp_rows.masked_fill(is_let, float("-inf")), dim=-1)   # log p(kết thúc)
        for w, s in zip(words, (base_scores + term).tolist()):
            if w and s > cands.get(w, float("-inf")):
                cands[w] = s

    logp = torch.log_softmax(logits.float(), dim=-1)                                  # (1, V)
    add_candidates([prefix], logp, torch.zeros(1, device=logp.device))                 # mode a: prefix đã là từ
    if max_word_len <= 0:
        return sorted(cands.items(), key=lambda kv: (-kv[1], kv[0]))[:k]

    # bước 1: nhánh từ trạng thái hiện tại (1 dòng)
    first = logp[0, letters]
    top = first.topk(min(beam, n_let))
    scores, toks = top.values, letters[top.indices]
    gens = [vocab.itos[t] for t in toks.tolist()]
    state_b = model.reorder_state(state, torch.zeros(len(gens), dtype=torch.long, device=logits.device))
    logits_b, state_b = model.step(toks, state_b)

    for length in range(1, max_word_len + 1):
        logp_b = torch.log_softmax(logits_b.float(), dim=-1)                          # (nb, V)
        add_candidates([prefix + g for g in gens], logp_b, scores)
        if length == max_word_len:
            break
        if len(cands) >= k:                                                            # dừng sớm chính xác
            kth = sorted(cands.values(), reverse=True)[k - 1]
            if kth > scores.max().item():
                break
        total = (scores[:, None] + logp_b[:, letters]).flatten()                       # (nb · n_let)
        top = total.topk(min(beam, total.numel()))
        parent, li = top.indices // n_let, top.indices % n_let
        toks = letters[li]
        scores = top.values
        gens = [gens[p] + vocab.itos[t] for p, t in zip(parent.tolist(), toks.tolist())]
        state_b = model.reorder_state(state_b, parent)
        logits_b, state_b = model.step(toks, state_b)

    return sorted(cands.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


def suggest(model, vocab, typed_text: str, k: int = 3, beam: int = 5, max_word_len: int = 10,
            l_ctx: int | None = None, return_scores: bool = False):
    """3 từ gợi ý cho toàn bộ văn bản người dùng đã gõ (NFC, giữ chữ hoa).
    l_ctx: số ký tự cuối model được nhìn (eval_spec mục 0; mặc định = toàn bộ văn bản)."""
    text = normalize(typed_text)
    ctx = TypingContext(model, vocab, l_ctx or max(1, len(text)), text)
    logits, state = ctx.current()
    out = suggest_from_state(model, vocab, logits, state, split_prefix(text), k, beam, max_word_len)
    return out if return_scores else [w for w, _ in out]
