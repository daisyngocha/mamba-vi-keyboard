"""Chuẩn bị dữ liệu Wikipedia tiếng Việt. Owner: Ngô Thị Thu Hằng.

Chạy:  python data/prepare_data.py --out data/processed --max_mb 80

Các bước cần cài đặt (xem docs/scope.md mục 3):
1. Tải HF dataset `wikimedia/wikipedia`, config `20231101.vi` (streaming=True để không tải hết).
2. Với mỗi bài: CharVocab.normalize (NFC + lowercase), bỏ dòng < 20 ký tự, gộp khoảng trắng.
3. Dừng khi tổng dung lượng text đạt --max_mb.
4. Chia theo BÀI VIẾT: train 90 / val 5 / test 5, seed 42.
5. Vocab: ký tự xuất hiện >= 50 lần trong train; id 0 = "<unk>". Lưu vocab.json.
6. Encode -> uint16 -> train.bin / val.bin / test.bin; ghi stats.json.
7. Sinh tập eval gợi ý từ: data/processed/wordsuggest_test.jsonl
   mỗi dòng {"context": "...", "prefix": "n", "target": "nắng", "k": 1}, 2.000 dòng từ test split,
   context 64–256 ký tự kết thúc ở ranh giới từ, k ∈ {0,1,2}, target là từ có >= 2 ký tự.

Kiểm tra trước khi mở PR: in 5 mẫu decode lại từ train.bin; tỉ lệ <unk> < 0.1%;
đếm NFD-only chars = 0.
"""
raise SystemExit("TODO(Thu Hằng): cài đặt prepare_data.py")
