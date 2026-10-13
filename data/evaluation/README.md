# Evaluation datasets

- `legacy.jsonl`: 15 câu thông tục đang hoạt động trong benchmark cũ, dùng làm hồi quy/dev.
- `dev.jsonl`: 107 câu khôi phục từ các bộ câu hỏi đã được comment trong mã cũ.
- `test.jsonl`: 35 câu cùng nguồn, giữ riêng theo điều luật. Các câu cùng điều không nằm cả dev và test.
- `edge_cases.jsonl`: câu ngoài phạm vi, thiếu dữ kiện và kiểm tra bằng chứng chi tiết.

Nhãn `legacy_unreviewed` chưa được chuyên gia pháp lý rà soát độc lập. Test được tách để tránh tinh chỉnh trực tiếp trên kết quả, nhưng xuất phát từ cùng repository nên chưa phải tập kiểm định độc lập bên ngoài.

`expected_articles` đo ở cấp điều. `expected_evidence_groups` là danh sách nhóm căn cứ bắt buộc: phải đáp ứng mỗi nhóm; các selector trong một nhóm là lựa chọn thay thế. Selector có thể gồm `document_id`, `article`, `clause`, `point`, `document_version`.

Không suy ra độ đúng câu trả lời từ Hit Rate, source ID hợp lệ hoặc quote có trong context. Báo cáo đầu cuối lưu câu trả lời và nguồn để rà soát ngữ nghĩa; điểm chưa chấm phải để trống.

Sau khi dùng test để chọn tham số/prompt, chuyển nó thành dev và lập tập test mới cho đánh giá cuối.
