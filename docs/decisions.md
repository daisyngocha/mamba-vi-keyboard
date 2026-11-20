# Nhật ký quyết định kỹ thuật

Tài liệu gốc về phạm vi: *Mô tả bài toán thực nghiệm (v1)* và *Hướng dẫn công việc Tuần 1 theo vai trò (v1)*. Repo đi theo hai tài liệu này; `docs/scope.md` (v0.1) đã bỏ. Quyết định mới hoặc thay đổi so với tài liệu thì ghi vào đây qua PR.

| # | Ngày | Quyết định | Lý do / ghi chú |
|---|---|---|---|
| Q1 | 19/11 | Báo cáo buổi 2 ngày 26/12/2026, 30' + 15' Q&A, chỉ trình bày thực nghiệm. 4 tuần 20/11–17/12, 18–25/12 dự phòng | Mô tả bài toán mục 5 |
| Q2 | 19/11 | **Giữ chữ hoa** (chỉ chuẩn hóa NFC), từ vựng ≈ 200–300 ký tự phủ ≥ 99,9%, lưu uint16 | Đúng tài liệu mục 2; bỏ đề xuất chuyển chữ thường của scope v0.1. Hệ quả: gợi ý đúng khi khớp cả hoa/thường — ĐG chốt chi tiết trong `docs/eval_spec.md` |
| Q3 | 19/11 | Chỉ dùng Wikipedia tiếng Việt | Văn phong khác tin nhắn: nêu là hạn chế |
| Q4 | 19/11 | Không làm Selective Copying | Ngoài phạm vi tài liệu v1; dồn thời gian cho RQ1–RQ3 |
| Q5 | 19/11 | Hai model dùng chung `models/base.py` (embedding khởi tạo std 0.02, RMSNorm cuối, lm_head dùng chung trọng số với embedding) — chỉ khác lõi | Mô tả bài toán mục 3: "chỉ khác phần lõi" |
| Q6 | 19/11 | `base.yaml` [đề xuất]: d_model 384, batch 32.768 ký tự/bước, 200M ký tự huấn luyện, lr 1e-3 cosine, warmup 2% | Chốt ở N7 sau khi MB/TF đếm tham số thật và đo bộ nhớ |
| Q7 | 20/11 (sớm hơn N4) | **Backend Mamba: `cuda`** — mamba-ssm 2.3.2.post1 + causal-conv1d 1.7.0, tự build trên Kaggle, wheel + 9 gói phụ thuộc lưu ở Dataset `mamba-btl-wheels` | Môi trường: torch 2.11.0+cu130, CUDA 13.0, Python 3.13.16, Tesla T4. Smoke test: `selective_scan_fn` lệch `selective_scan_ref` 6,5e-05; forward + backward OK. **Mọi notebook phải chọn Environment = Pin to original environment (2026-11-19)**, nếu không wheel có thể không khớp torch |
| Q8 | 20/11 (xác nhận ở N7) | **fp16** (autocast + GradScaler). d_model: Transformer 384 (6×64), Mamba 368 → 7,19M / 7,20M tham số | 500 bước, L = 256, T4: Mamba fp16 vs fp32 cho loss và val BPC trùng tới 4 chữ số (2,0296 / 2,0295), fp16 nhanh gấp 2 (53k / 26k ký tự/s) và ít bộ nhớ hơn (2,7 / 4,4 GB). Không NaN. L tối đa của Transformer: chờ số đo L = 4.096 |
