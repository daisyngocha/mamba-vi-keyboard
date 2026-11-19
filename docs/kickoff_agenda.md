# Kickoff — Thứ 5 19/11/2026, 20:30–21:30 (Google Meet)

Chủ trì: Lê Thị Ngọc Hà · Thư ký ghi decision log: **[chỉ định 1 người]**
Chuẩn bị: mọi người đã đọc `docs/scope.md`; lead share màn hình scope + repo.

## Agenda
| Giờ | Nội dung | Kết quả cần có |
|---|---|---|
| 20:30–20:35 | Mục tiêu BTL, hạn nộp, cách chấm (đọc lại yêu cầu của thầy) | Cả nhóm hiểu giống nhau |
| 20:35–20:50 | Walkthrough scope: bài toán, dữ liệu, mô hình, metric | Câu hỏi/góp ý ghi vào log |
| 20:50–21:05 | Thông báo 7 quyết định đã chốt (D1–D7, bảng dưới), ghi nhận phản đối nếu có | Cả nhóm xác nhận |
| 21:05–21:15 | Phân công tuần 1 + giải thích interface chung (`forward(idx)->logits`, config YAML, `metrics.csv`) | Mỗi người nhận đúng 1 issue chính |
| 21:15–21:22 | Quy ước Git, daily update, lịch sync, quota GPU | Đồng ý quy ước |
| 21:22–21:30 | Chốt deadline tuần 1, rủi ro, ai đang bị chặn | Mỗi người nói lại task + deadline của mình |

Mẹo điều phối: D1–D7 đã được trưởng nhóm chốt trước họp; mỗi phản đối thảo luận tối đa 2 phút, nếu cần đổi thì mở PR sửa scope và xem lại ở sync Thứ 2.

## Decision log
| # | Vấn đề | Đề xuất của lead | Quyết định | Ghi chú |
|---|---|---|---|---|
| D1 | Ngày + thời lượng buổi báo cáo 2 (chỉ thực nghiệm) | 26/12/2026, 30' + 15' Q&A | Đồng ý — chốt 19/11 | |
| D2 | Thêm corpus tin tức ngoài Wikipedia? | Không ở tuần 1; cân nhắc tuần 2 | Đồng ý — chốt 19/11 | |
| D3 | Chuyển chữ thường? | Có | Đồng ý — chốt 19/11 | |
| D4 | Quét lr cho Transformer | {3e-4, 1e-3} | Đồng ý — chốt 19/11 | |
| D5 | Ngân sách train | 20k bước, seq_len 256, batch 64 | Đồng ý — chốt 19/11 | |
| D6 | Phân công tuần 1 | Theo scope mục 9 | Đồng ý — chốt 19/11 | |
| D7 | Công cụ log | CSV trong `results/` (W&B tuỳ chọn) | Đồng ý — chốt 19/11 | |

## Action items (điền trong họp)
| Việc | Owner | Deadline |
|---|---|---|
| | | |

## Tin nhắn tóm tắt gửi nhóm sau họp (mẫu)
> Tóm tắt kickoff 19/11:
> • Đã chốt: D1…D7 (chi tiết trong docs/kickoff_agenda.md trên repo).
> • Tuần 1 (20–26/11): mỗi người làm issue của mình trên GitHub, PR trước **CN 23/11 22h**.
> • Sync: **Thứ 2 24/11 21:00** (ghép pipeline) và **Thứ 4 26/11 21:00** (tổng kết tuần 1).
> • Daily update trước 22h mỗi tối: Làm gì · Bị chặn · Mai làm gì.
