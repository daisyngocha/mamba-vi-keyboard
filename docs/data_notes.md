# Ghi chú số liệu dữ liệu (cho chương Dữ liệu)

Nguồn số liệu: `meta.json`, `survey/survey.json`, `survey/length_hist.png` của Kaggle Dataset `mamba-btl-data` (tạo ngày 20/11/2026 bằng `python -m data.prepare`, seed 0, chạy 114 giây trên notebook CPU Kaggle).

## 1. Nguồn và giấy phép
- HuggingFace `wikimedia/wikipedia`, cấu hình `20231101.vi` (bản dump Wikipedia tiếng Việt ngày 01/11/2023, văn bản thuần), thư viện `datasets` 4.8.5, đọc dạng streaming, trộn bằng bộ đệm 10.000 bài theo seed 0.
- Giấy phép CC BY-SA: phải ghi nguồn trong báo cáo và slide (`docs/references.md`).

## 2. Khảo sát ban đầu (5.000 bài đầu luồng, trước khi lọc)
| Chỉ số | Giá trị |
|---|---|
| Độ dài trung vị | 178 ký tự |
| Phân vị 10 / 25 / 75 / 90 / 99 | 138 / 146 / 191 / 203 / 253 ký tự |
| Độ dài trung bình | 175 ký tự |
| Tỷ lệ bài < 2.000 ký tự | **99,94%** (chiếm 98,9% tổng số ký tự) |
| Số ký tự khác nhau | 188 |
| Số ký tự cần để phủ 99% / 99,9% / 99,99% | 69 / 106 / 147 |

Hình `figures/length_hist_survey.png`: phân bố log10 độ dài có hai đỉnh hẹp quanh ~140 và ~190 ký tự, gần như không có bài nào vượt ngưỡng 2.000 (đường đỏ). Đây là dấu hiệu của các bài sơ khai do bot tạo theo cùng một khuôn mẫu (trên Wikipedia tiếng Việt, loại bài này phần lớn viết về các loài sinh vật). Đó là lý do lọc bỏ bài dưới 2.000 ký tự: nếu giữ lại, model chủ yếu học thuộc vài khuôn câu lặp lại thay vì văn bản tự nhiên.

## 3. Lọc và thu thập
| Bước | Số bài |
|---|---|
| Đã duyệt | 318.533 |
| Bỏ vì < 2.000 ký tự | 312.033 (98,0%) |
| Bỏ vì trùng nội dung | 0 |
| **Giữ lại** | **6.500** (2,0%), tổng 53,06M ký tự, trung bình ≈ 8.160 ký tự/bài |

Làm sạch mỗi bài: Unicode NFC (tránh một chữ có dấu biểu diễn theo hai cách), bỏ ký tự vô hình (zero-width, soft hyphen), gộp khoảng trắng, bỏ dòng trống. **Giữ chữ hoa** (quyết định Q2).

## 4. Chia tập (theo bài viết, xáo bằng seed 0)
| Tập | Số ký tự | Số bài | Tỷ lệ `<unk>` |
|---|---|---|---|
| Train | 50.011.918 | 6.137 | 0,100% |
| Validation | 1.000.567 | 120 | 0,058% |
| Test | 1.002.118 | 107 | 0,133% |
| Sample (1M ký tự đầu train, để thử code) | 1.000.000 | | |

Mỗi bài thuộc đúng một tập (đã kiểm tra trong `prepare.py`, danh sách id ở `splits.json`), nên không có rò rỉ nội dung giữa train và test. Số ký tự tính cả ký tự `<doc>` mở đầu mỗi bài.

## 5. Từ vựng mức ký tự
- Tập train có **5.932 ký tự khác nhau**: chữ Latin, chữ Việt có dấu, cùng chữ Hán, Nhật, Hy Lạp, Cyrillic… xuất hiện rải rác trong tên riêng và trích dẫn.
- Giữ **273 ký tự phổ biến nhất**, phủ 99,90% số ký tự train; thêm `<unk>` (id 0) và `<doc>` (id 1) thành **vocab_size = 275**. Vì vocab > 255 nên file `.bin` lưu dạng uint16.
- Ví dụ ký tự bị gộp vào `<unk>`: Ạ λ 林 ë б κ 張 イ の 安 帝 金 ¥ К х … Đa số là ký tự ngoại ngữ. Riêng **"Ạ"** (chữ A hoa có dấu nặng) cũng nằm ngoài từ vựng vì chữ hoa có dấu rất hiếm trong văn bản, đúng như dự đoán khi chọn giữ chữ hoa.
- So với khảo sát (106 ký tự đủ phủ 99,9%), từ vựng thật lớn hơn nhiều vì bài dài có nội dung đa dạng hơn bài sơ khai.

## 6. Tập mô phỏng gõ phím (`typing_sim.jsonl`)
2.000 vị trí ngẫu nhiên trong tập test: 1.000 cho chức năng (a) hoàn thành từ đang gõ, 1.000 cho (b) gợi ý từ tiếp theo. Mỗi vị trí giữ tối đa 4.096 ký tự văn bản phía trước. Định dạng và cách chấm: `docs/eval_spec.md` mục 3.

## 7. Hạn chế cần nêu trong báo cáo
1. **Văn phong**: Wikipedia là văn viết bách khoa, khác văn phong tin nhắn trên bàn phím điện thoại (Mô tả bài toán mục 8).
2. **Tập val/test ít bài** (120 và 107 bài, mỗi bài trung bình 8–9 nghìn ký tự): chỉ số đánh giá có thể chịu ảnh hưởng của chủ đề vài bài dài. Báo cáo nên kèm độ lệch giữa các seed tại L = 1.024.
3. **Tỷ lệ `<unk>` trên test (0,133%)** cao hơn val (0,058%) và vượt nhẹ ngưỡng 0,1% đặt cho val. Ảnh hưởng tới BPC rất nhỏ. Từ chứa ký tự `<unk>` không thể được gợi ý đúng.
4. **Đại diện**: chỉ duyệt 318.533 bài (≈ 25% bản dump) trước khi đủ 53M ký tự. Bộ đệm trộn 10.000 bài không trộn đều toàn bộ dump, nên phân bố chủ đề có thể lệch theo các shard được đọc trước.
5. Phần khảo sát chỉ phản ánh 5.000 bài đầu luồng, phần lớn là bài sơ khai, nên **không** mô tả phân bố độ dài của 6.500 bài được giữ lại. Nếu chương Dữ liệu cần hình độ dài bài sau lọc, tính thêm từ `splits.json` hoặc đếm lại ranh giới `<doc>` trong `train.bin`.
