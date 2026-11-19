"""Đánh giá chất lượng. Owner: Cao Thanh Hằng.

Chạy:  python -m src.evaluate --ckpt results/<run>/last.pt --split test

Cần cài đặt:
1. test_bpc(model, ds): loss trung bình trên toàn bộ test.bin (cửa sổ không chồng lấn, seq_len như lúc train).
2. suggest_words(model, vocab, context, prefix, k=3, beam=5, max_len=20) -> list[str]
   - Encode context + prefix, beam search ký tự đến khi gặp " " hoặc dấu câu.
   - Gộp từ trùng, xếp theo tổng log-prob.
3. word_topk_accuracy(model, vocab, jsonl_path) -> {"top1@k0", "top3@k0", "top1@k1", ..., "top3@k2"}
   đọc data/processed/wordsuggest_test.jsonl (contract với Thu Hằng).
4. keystroke_savings_rate(...): mô phỏng gõ từng ký tự, nếu target nằm trong top-3 thì "chọn" và tính số phím tiết kiệm.
5. Ghi kết quả ra results/<run>/eval_<split>.json.

Kiểm tra: chạy với checkpoint smoke test (results/smoke/last.pt) phải ra số, không lỗi.
"""
raise SystemExit("TODO(Thanh Hằng): cài đặt evaluate.py")
