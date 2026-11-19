"""Chuẩn bị dữ liệu Wikipedia tiếng Việt. Owner: DL (Ngô Thị Thu Hằng).

Chạy:  python data/prepare.py --out data/processed --seed 0
Sau đó upload thư mục --out lên Kaggle Dataset `mamba-btl-data` (private, nhóm là collaborator).

Đặc tả — Mô tả bài toán mục 2, Hướng dẫn Tuần 1 mục 5:
1. Tải HF `wikimedia/wikipedia`, cấu hình `20231101.vi`, split train, streaming=True.
2. Khảo sát ≈ 5.000 bài: phân bố độ dài, tỷ lệ bài < 2.000 ký tự, tập ký tự và tần suất
   -> lưu bảng/biểu đồ cho chương Dữ liệu.
3. Làm sạch từng bài: Unicode NFC (data.loader.normalize); gộp khoảng trắng và dòng trống thừa;
   BỎ BÀI < 2.000 ký tự (phần lớn là bài sơ khai do bot tạo). GIỮ CHỮ HOA.
4. Chia theo BÀI VIẾT: xáo danh sách bài bằng seed cố định, lần lượt gán test (≈ 1M ký tự),
   val (≈ 1M), train (≈ 50M). Một bài chỉ thuộc một tập.
5. Từ vựng xây TỪ TẬP TRAIN: giữ các ký tự phổ biến nhất sao cho phủ ≥ 99,9% số ký tự (≈ 200–300 ký tự,
   gồm chữ hoa/thường có dấu); còn lại -> <unk>. id 0 = <unk>, id 1 = <doc>. Lưu vocab.json (data.loader.CharVocab).
6. Mã hóa: mỗi bài mở đầu bằng <doc>, nối liền, lưu train.bin / val.bin / test.bin dạng uint16.
   meta.json: vocab_size, n_chars mỗi tập, unk_rate mỗi tập, dataset "wikimedia/wikipedia 20231101.vi",
   min_article_chars 2000, coverage 0.999, seed.
7. sample.bin: ≈ 1M ký tự đầu của train.bin — làm TRƯỚC (N2) để mọi người thử code.
8. (Nếu kịp, có thể dời sang tuần 2) tập mô phỏng gõ phím typing_sim.jsonl: ≈ 2.000 vị trí ngẫu nhiên trong
   test; phần văn bản phía trước là "người dùng đã gõ", từ thực tế phía sau là đáp án. Thống nhất định dạng
   với ĐG theo đặc tả đánh giá (docs/eval_spec.md).

Kiểm tra trước khi mở PR: in 5 đoạn decode lại từ train.bin; tỷ lệ <unk> trên val < 0,1%;
không có bài trùng giữa các tập; số ký tự NFD-only = 0.
"""
raise SystemExit("TODO(DL): cài đặt data/prepare.py")
