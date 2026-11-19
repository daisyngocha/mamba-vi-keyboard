# Dàn ý slide — Mamba: Linear-Time Sequence Modeling with Selective State Spaces

Thời lượng: 60 phút trình bày + 20 phút Q&A · Template: HUST · Người trình bày: Lê Thị Ngọc Hà
Mục tiêu ~40 slide (≈ 1,5 phút/slide). Mỗi phần có 1 người soạn nháp; lead thống nhất văn phong và sơ đồ.
Hạn nháp: phần lý thuyết (1–8) **10/12**, phần ứng dụng (9) **12/12** (cần số liệu tuần 2–3).

| # | Phần | Phút | Slide | Soạn nháp | Sơ đồ có sẵn (project) |
|---|---|---|---|---|---|
| 1 | Mở đầu | 2 | 2 | Ngọc Hà | — |
| 2 | Bối cảnh: RNN vs Transformer | 6 | 4 | Duyên Hà | `RNN_Architecture`, `Transformer_Architecture` |
| 3 | Nền tảng SSM → S4 | 9 | 6 | Ngọc Hà | `SSM_Architecture_pre_Mamba` |
| 4 | Hạn chế của LTI: Selective Copying, Induction Heads | 5 | 3 | Thanh Hằng | `Table1_SelectiveCopying_Architectures` |
| 5 | Selection mechanism (S6) | 9 | 6 | Tiến Dũng | `Mamba_Block_and_Scan` |
| 6 | Thuật toán hardware-aware (selective scan) | 6 | 4 | Tiến Dũng | `Mamba_Block_and_Scan` |
| 7 | Kiến trúc Mamba block | 4 | 3 | Tiến Dũng | `Mamba_Architecture` |
| 8 | Kết quả trong paper | 4 | 3 | Ngọc Hà | Bảng/hình gốc của paper (ghi nguồn) |
| 9 | Ứng dụng: gợi ý từ bàn phím tiếng Việt | 12 | 8 | Thu Hằng (9a–b) · Thanh Hằng (9c–e) | Vẽ mới |
| 10 | Thảo luận, hạn chế, kết luận | 3 | 2 | Ngọc Hà | — |
| | **Tổng** | **60** | **41** | | |

## Nội dung chính từng phần

**1. Mở đầu** — Tên paper, tác giả (Albert Gu, Tri Dao), thành viên nhóm; agenda; 1 câu thông điệp: *Mamba đạt chất lượng ngang Transformer với chi phí tuyến tính theo độ dài chuỗi nhờ SSM có chọn lọc*.

**2. Bối cảnh** — Sequence modeling; RNN: suy luận O(1)/bước nhưng train tuần tự, khó nhớ xa; Transformer: train song song nhưng attention O(L²), KV cache tăng theo L; câu hỏi nghiên cứu: có mô hình nào vừa train song song, vừa suy luận tuyến tính, vừa mạnh như attention không.

**3. SSM → S4** — SSM liên tục (A, B, C, D); rời rạc hoá với Δ (ZOH); 2 góc nhìn: recurrent (suy luận) và convolution (train); S4 và họ SSM có cấu trúc; tính chất LTI (tham số không phụ thuộc input).

**4. Hạn chế của LTI** — Selective Copying vs Copying; Induction Heads; vì sao convolution/LTI không làm được (không lọc được theo nội dung); trade-off hiệu quả và hiệu năng: kích thước state càng nén thì càng hiệu quả, nhưng phải nén *có chọn lọc*.

**5. S6 — Selection** — Δ, B, C trở thành hàm của input (Algorithm 2 so với Algorithm 1); diễn giải Δ: Δ lớn → reset và tập trung vào input hiện tại, Δ nhỏ → bỏ qua input; liên hệ cơ chế gating của RNN (Theorem 1); cái giá: mất góc nhìn convolution → cần thuật toán mới.

**6. Hardware-aware** — Phân cấp bộ nhớ GPU (HBM so với SRAM); kernel fusion: nạp tham số vào SRAM, rời rạc hoá + scan + nhân C trong SRAM; parallel scan; recomputation khi backward để tiết kiệm bộ nhớ; kết quả: nhanh hơn FlashAttention-2 ở chuỗi dài.

**7. Mamba block** — Gộp H3 + gated MLP thành 1 block đồng nhất; conv1d, SiLU, nhánh gate; xếp chồng block, RMSNorm, residual; số tham số ~3·expand·d².

**8. Kết quả paper** — Synthetic (Selective Copying, Induction Heads ngoại suy đến 1M token); scaling LM so với Transformer++; DNA, audio; throughput suy luận gấp ~5× Transformer.

**9. Ứng dụng** (nhóm tự làm)
- 9a. Bài toán và demo trải nghiệm bàn phím (1 slide)
- 9b. Dữ liệu và tiền xử lý: Wiki-vi, NFC, vocab ký tự, chia theo bài viết (1 slide)
- 9c. Setup so sánh công bằng: bảng cấu hình, số tham số, ngân sách train (1 slide)
- 9d. Kết quả: E1 Selective Copying, E2 BPC + Top-k + KSR, E3 thời gian/bộ nhớ theo seq_len (3–4 slide; E3 là biểu đồ quan trọng nhất)
- 9e. Demo trực tiếp hoặc video 30 giây (1 slide)

**10. Kết luận** — Điều nhóm xác nhận được và chưa xác nhận được so với paper; hạn chế (mô hình nhỏ, chưa có kernel CUDA nếu không cài được mamba-ssm, dữ liệu Wikipedia ≠ văn gõ hằng ngày); hướng mở rộng (Telex, Mamba-2, hybrid).

## Danh sách chuẩn bị Q&A (mỗi người soạn câu trả lời ≤ 5 dòng cho câu được giao)
| Câu hỏi dự kiến | Người soạn |
|---|---|
| Vì sao selection làm mất tính LTI, và vì sao điều đó buộc phải dùng scan thay cho convolution? | Tiến Dũng |
| Δ đóng vai trò gì? Vì sao A không cần phụ thuộc input? | Tiến Dũng |
| So sánh độ phức tạp train/suy luận của Mamba, Transformer, RNN | Duyên Hà |
| Parallel scan hoạt động thế nào? Vì sao recomputation lại tiết kiệm bộ nhớ? | Ngọc Hà |
| Mamba có điểm yếu gì so với attention (recall chính xác, in-context learning)? | Ngọc Hà |
| Vì sao chọn char-level? Vì sao không dùng tokenizer BPE? | Thu Hằng |
| So sánh có công bằng không (số tham số, learning rate, số bước)? | Thanh Hằng |
| Bản tự cài đặt chậm hơn mamba-ssm bao nhiêu, vì sao? | Tiến Dũng |
| Nếu Transformer thắng về BPC thì kết luận gì? | Ngọc Hà |
| Bài toán này có triển khai thực tế được trên điện thoại không (latency, kích thước)? | Thanh Hằng |
