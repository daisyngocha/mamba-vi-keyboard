"""Hàm gợi ý top-3 từ. Owner: ĐG (Ngô Thị Thanh Hằng). Tuần 2.

Đặc tả — Mô tả bài toán mục 1, 4.2, 8:
    suggest(model, vocab, typed_text, k=3, beam=5, max_word_len=10) -> list[str]
- typed_text là toàn bộ văn bản người dùng đã gõ (NFC, GIỮ CHỮ HOA).
  (a) Hoàn thành từ đang gõ: typed_text không kết thúc bằng dấu cách -> phần sau dấu cách cuối là tiền tố,
      gợi ý là từ hoàn chỉnh bắt đầu bằng tiền tố đó ("Hôm nay trời đ" -> đẹp · đang · đã).
  (b) Gợi ý từ tiếp theo: typed_text kết thúc bằng dấu cách ("Hôm nay trời␣" -> mưa · nắng · đẹp).
- Beam search mức ký tự: sinh tiếp đến dấu cách/dấu câu; beam 3–5, tối đa ≈ 10 ký tự mỗi từ; "từ" = âm tiết.
- Gộp ứng viên trùng, xếp theo tổng log-prob, trả về 3 từ.
- Tuần 2 dùng model.step(token, state) để không phải tính lại cả chuỗi mỗi ký tự.
Định nghĩa chính xác Top-1/Top-3 nằm trong docs/eval_spec.md (ĐG soạn tuần 1).
"""


def suggest(model, vocab, typed_text: str, k: int = 3, beam: int = 5, max_word_len: int = 10) -> list[str]:
    raise NotImplementedError("TODO(ĐG): beam search mức ký tự, tuần 2")
