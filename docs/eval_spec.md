# Đặc tả đánh giá (Hướng dẫn Tuần 1 mục 8 · Mô tả bài toán mục 4.2–4.3)

Owner: ĐG. Mọi chỉ số trong báo cáo tính đúng theo trang này; thay đổi phải qua PR và ghi vào `docs/decisions.md`.
Áp dụng như nhau cho Mamba và Transformer: cùng checkpoint cuối của lượt chạy, cùng tập, cùng cấu hình gợi ý.

## 0. Quy ước chung
- **Văn bản**: NFC, **giữ chữ hoa** (Q2). Văn bản người dùng gõ cũng qua NFC trước khi encode.
- **Từ** = âm tiết gồm toàn chữ cái (`str.isalpha()`), tách theo khoảng trắng. Ký tự kết thúc từ = mọi ký tự không phải chữ cái (khoảng trắng, dấu câu, xuống dòng, chữ số).
- **Ngữ cảnh**: model nhìn tối đa `L_ctx` ký tự cuối của văn bản đã gõ; mặc định `L_ctx = L` lúc huấn luyện. Thí nghiệm ngoại suy đổi `L_ctx` (mục 6).
- **Cấu hình gợi ý cố định**: beam 5, sinh tối đa 10 ký tự mỗi từ, trả 3 từ. GPU T4, fp16, batch 1 (trừ BPC).
- **Báo cáo**: tại L = 1.024 báo mean ± std trên 3 seed; L khác 1 seed. Kết quả mỗi lượt ghi `logs/<run_name>_eval.json`.

## 1. BPC (chất lượng mô hình)
- Trên **toàn bộ `test.bin`** (≈ 1M ký tự), cắt thành các cửa sổ **không chồng nhau** dài L (bỏ phần dư cuối), mọi vị trí đều được chấm.
- `BPC = (Σ −ln p(x_t | x_<t)) / (N · ln 2)`, N = số ký tự được chấm (kể cả `<doc>`, `<unk>`).
- Val BPC trong lúc train (20 cửa sổ ngẫu nhiên cố định) chỉ để theo dõi, **không** đưa vào bảng kết quả.

## 2. Hàm gợi ý `suggest(model, vocab, typed, k=3)`
- **(a) hoàn thành từ**: `typed` kết thúc giữa từ → `prefix` = phần chữ cái sau ký tự kết thúc cuối cùng. Ứng viên = `prefix + phần sinh thêm` (phần sinh thêm có thể rỗng nếu `prefix` đã là một từ).
- **(b) từ tiếp theo**: `typed` kết thúc bằng ký tự kết thúc → `prefix = ""`, ứng viên phải có ≥ 1 chữ cái.
- **Điểm** của ứng viên `w` = `Σ log p(ký tự sinh thêm) + log p(kết thúc | typed + w)`, trong đó `p(kết thúc)` = tổng xác suất các ký tự không phải chữ cái. Có hạng tử kết thúc để không ưu tiên từ cụt.
- Beam search theo từng ký tự, chỉ mở rộng bằng chữ cái; giữ 5 nhánh; dừng ở 10 ký tự. Lấy 3 ứng viên **khác nhau** điểm cao nhất (hòa điểm → thứ tự từ điển).

## 3. Top-1 / Top-3 trên `typing_sim.jsonl`
- Tập mô phỏng do `data/prepare.py` sinh: 2.000 vị trí ngẫu nhiên trong test (1.000 mode a, 1.000 mode b), mỗi dòng `{mode, typed, prefix, target}`.
- **Đúng** khi `target` khớp **chính xác** (cả hoa/thường) với gợi ý thứ 1 (Top-1) hoặc một trong 3 gợi ý (Top-3).
- Báo riêng (a), (b) và gộp. Chỉ số phụ: Top-3 **không phân biệt hoa/thường** (để tách lỗi viết hoa khỏi lỗi chọn từ).

## 4. KSR — tỷ lệ tiết kiệm phím gõ
- Lấy **50 đoạn × 400 ký tự** từ test (seed 0, cắt ở ranh giới từ). Gõ lại từng đoạn, từng ký tự.
- Trước mỗi phím trong một từ (kể cả trước chữ cái đầu tiên), lấy 3 gợi ý. Nếu từ đúng có trong 3 gợi ý thì **chạm chọn**: tốn 1 phím, chèn phần còn lại của từ (**không** tự thêm khoảng trắng), rồi gõ tiếp ký tự sau từ.
- Ký tự không phải chữ cái luôn gõ tay.
- `KSR = (tổng số ký tự − tổng số phím thực bấm) / tổng số ký tự`. Gõ không có gợi ý thì KSR = 0.

## 5. Độ trễ và bộ nhớ khi gợi ý (RQ3)
- Văn bản đã gõ dài `T ∈ {256, 512, 1.024, 2.048, 4.096, 8.192}` ký tự (lấy từ test). **Nạp trước** T ký tự bằng `model.step()` (Mamba: trạng thái cố định; Transformer: KV cache). Thời gian nạp không tính.
- **Độ trễ 1 phím gõ** = thời gian `step(ký tự mới)` + `suggest` (beam 5, ≤ 10 ký tự). Đo bằng `bench.measure`: 5 lần khởi động, trung vị của 50 phím liên tiếp, batch 1. Báo trên GPU T4; tùy chọn CPU fp32 1 luồng (gần điện thoại).
- **Bộ nhớ**: `peak_mem_mb` trong lúc gợi ý và kích thước trạng thái tính bằng công thức (fp16):
  - Transformer KV cache = `2 · n_layer · T · d_model · 2 byte` (≈ 6 KB/ký tự với 4 × 384)
  - Mamba = `n_layer · d_inner · (d_state + d_conv − 1) · 2 byte`, không phụ thuộc T
- **Kích thước model** = số tham số × 2 byte (fp16), báo MB.
- **Điều kiện đúng** (test tuần 2): logits khi sinh bằng `step()` phải bằng logits khi chạy `forward` trên cả chuỗi, sai số ≤ 1e-3.
- Transformer chỉ được train tới L = 4.096, nên với T > L_ctx thì cắt ngữ cảnh còn `L_ctx`. Ghi rõ ràng buộc này trong báo cáo; độ trễ và bộ nhớ vẫn đo với cache dài T.

## 6. Thí nghiệm khám phá (Mô tả bài toán mục 4.3)
- **Ngữ cảnh dài có giúp?** Top-3 và KSR theo L ∈ {256, 1.024, 4.096}, `L_ctx = L`.
- **Ngoại suy độ dài**: model train ở L = 1.024; chấm Top-3 với `L_ctx ∈ {1.024, 2.048, 4.096}` (dùng phần `typed` dài hơn; `typing_sim` có ngữ cảnh tới 4.096 ký tự).
- **Lỗi đặc thù tiếng Việt** (trên các ca Top-1 sai): so gợi ý số 1 với `target` sau khi
  1. bỏ 5 dấu thanh (U+0300 huyền, U+0301 sắc, U+0303 ngã, U+0309 hỏi, U+0323 nặng): nếu giống → **sai dấu thanh** (vd “đa” thay vì “đã”)
  2. bỏ mọi dấu phụ (kể cả ^, ˘, móc, đ → d): nếu giống → **sai dấu chữ**
  3. nếu chỉ khác hoa/thường → **sai viết hoa**
  4. còn lại → **sai chữ cái gốc**

## 7. Tốc độ huấn luyện (RQ2)
- `bench.py`: bước huấn luyện đầy đủ (forward + backward + AdamW, fp16), token ngẫu nhiên, B = 32.768 // L, trung vị 10 lần sau 3 lần khởi động → ký tự/giây, ms/bước, bộ nhớ đỉnh, OOM.
- Bổ sung từ log `train.py`: thời gian huấn luyện thực tế đến hết `train_chars` (`train_time_s` trong `_summary.json`).
- Mọi số đo tốc độ phải cùng loại GPU (T4) và ghi `gpu_name`.
