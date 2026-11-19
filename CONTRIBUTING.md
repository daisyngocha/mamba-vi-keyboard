# Quy ước làm việc

## Git
1. Không push thẳng vào `main`.
2. Mỗi issue một nhánh: `git checkout -b feat/<ten-ngan>` (vd `feat/mamba-pt`, `feat/data-wiki`).
3. Commit nhỏ, message dạng `model: add selective scan`, `data: NFC normalize`.
4. Mở Pull Request vào `main`, ghi `Closes #<số issue>`, mô tả: đã làm gì, chạy kiểm tra gì, kết quả (dán log/ảnh).
5. Trưởng nhóm review trong 24h. Sửa theo comment rồi push tiếp lên cùng nhánh.
6. Trước khi mở PR: `bash scripts/smoke_test.sh` phải in `SMOKE TEST OK`.

## Code
- Python 3.10+, type hint cho hàm public, docstring ngắn tiếng Việt hoặc Anh.
- Không hard-code tham số — đưa vào `configs/*.yaml`.
- Không commit dữ liệu, checkpoint (`*.pt`, `*.bin`), notebook có output lớn.

## Giao tiếp
- Daily update trên nhóm chat trước 22h: **Hôm nay làm gì · Bị chặn ở đâu · Mai làm gì**.
- Bị chặn > 2 giờ → nhắn ngay, không đợi đến buổi sync.
- Run GPU > 1 giờ → báo trước trong nhóm (quota Kaggle ~30h/tuần/tài khoản).
- Sync: Thứ 2 và Thứ 4, 21:00.
