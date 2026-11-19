# Quy ước làm việc (Hướng dẫn Tuần 1 mục 3.5)

## Git và review
1. Không ai push thẳng vào `main` (nhánh được bảo vệ, bắt buộc merge qua PR).
2. Mỗi việc một nhánh `<mã>/<việc>`: `dl/prepare`, `mb/mamba-lm`, `tf/transformer-lm`, `dg/train`, `tn/...`.
3. Commit nhỏ, message dạng `data: NFC normalize`, `model: add RoPE`.
4. Mở Pull Request vào `main`, ghi `Closes #<issue>`, mô tả ngắn: **làm gì · chạy thử thế nào · kết quả kiểm tra** (dán log/ảnh).
5. Trước khi mở PR: `bash scripts/smoke_test.sh` in `SMOKE TEST OK` và `python -m pytest -q tests` không fail.
6. TN review trong 24 giờ: đúng giao diện mục 3.2–3.4 (README.txt) và chạy được trên `notebooks/00_setup.ipynb`. Đạt thì merge; chưa đạt thì góp ý trên PR.
7. Lịch sử commit và PR là bằng chứng đóng góp của từng thành viên — commit bằng tài khoản GitHub của chính mình.

## Code
- Python 3.10+, type hint cho hàm public, docstring ngắn.
- Không hard-code tham số — đưa vào `configs/*.yaml`.
- Không commit dữ liệu, checkpoint, wheel (`*.pt`, `*.bin`, `*.whl`), notebook có output lớn. Log CSV cuối của lượt chạy thì commit vào `logs/`.
- Dùng thư viện/mã nguồn ngoài -> thêm ngay vào `docs/references.md`.

## Phối hợp
- Trễ một mốc mà người khác đang chờ (sample.bin N2, train.py N3, backend Mamba N4, dữ liệu đầy đủ N5) -> báo ngay trong nhóm chat.
- Ghi mỗi lượt chạy vào bảng lượt chạy; ghi phần việc và số giờ vào nhật ký đóng góp.
- Quota Kaggle ≈ 30 giờ GPU/tuần/tài khoản: lượt chạy > 1 giờ báo trước trong nhóm.
