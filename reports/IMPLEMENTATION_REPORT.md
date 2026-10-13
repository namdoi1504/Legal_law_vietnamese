# Kết quả cải thiện Legal RAG

Cập nhật ngày 13/10/2026. Phạm vi được thống nhất: snapshot Bộ luật Lao động 2019, đánh giá API tối đa 200 câu. Đây là báo cáo kỹ thuật; chưa phải chứng nhận chất lượng tư vấn pháp lý.

## Thay đổi đã triển khai

- Parser giữ phần dẫn của khoản khi tách điểm, nhận diện `đ)`, tiêu đề xuống dòng và phần trích dẫn sửa đổi tại Điều 219. Snapshot mới có **873 chunk, đủ 220 điều**, ID không trùng.
- BM25 và Qdrant dùng chung manifest/corpus version; UUID ổn định, kiểm tra hash và nội dung trước khi kích hoạt. Build trong thư mục riêng, giữ snapshot cũ để rollback. Qdrant local hoạt động không cần server.
- Retrieval hợp nhất câu gốc và câu viết lại; alias cho cách nói thông tục được thêm vào câu gốc. Rewrite không được đổi số hoặc bỏ phủ định đã nhận diện. Pool 40 candidates, rerank top 8, cho phép nhiều khoản cùng điều.
- Context mở rộng phần cha cần thiết, khử trùng lặp, giữ nguyên căn cứ trong ngân sách model. Filter được áp dụng ở cả BM25 và vector; dữ liệu chưa xác minh hiệu lực không được dùng để trả lời filter theo ngày.
- LLM trả JSON gồm nhận định và bằng chứng. Validator kiểm tra ID và đoạn trích nguyên văn; chỉ nguồn thực sự được trích dẫn xuất hiện trong câu trả lời. Có trạng thái thiếu thông tin, thiếu căn cứ, lỗi schema và lỗi provider.
- Rà soát PDF/DOCX truy xuất riêng từng điều khoản, giữ offset và nguồn; tổng hợp trực tiếp từ kết quả đã kiểm tra. Giới hạn dung lượng/số phần; báo phần chưa xử lý, dừng khi hết quota. PDF scan cần OCR; DOCX giữ thứ tự bảng.
- API dùng lifecycle chia sẻ model, threadpool và giới hạn concurrency; health/readiness, timeout, HTTP 429/Retry-After và HTTP 503. CI chạy kiểm thử giả lập, không gọi API trả phí.

## Số đo đã có

Kiểm chứng mã tại checkpoint: **44/44 unittest đạt**, cú pháp 48 file Python hợp lệ, `git diff --check` không phát hiện lỗi whitespace. Benchmark test35 vẫn đang chạy; trạng thái bàn giao và lệnh tiếp tục ở [PROGRESS.md](../PROGRESS.md).

Các tỷ lệ dưới đây là **truy xuất được điều luật mang nhãn trong top 8**, không phải tỷ lệ câu trả lời pháp lý đúng. Nhãn còn cần rà soát chuyên môn.

| Phép đo | Kết quả | Báo cáo |
|---|---:|---|
| BM25 corpus cũ, 15 câu hồi quy | 10/15 (66,7%) | `evaluation/baseline_legacy_bm25.json` |
| BM25 corpus mới, 15 câu hồi quy | 11/15 (73,3%) | `evaluation/final_legacy_bm25.json` |
| Pipeline trước alias, 15 câu hồi quy | 14/15 (93,3%) | `evaluation/current_legacy_pipeline.json` |
| Pipeline sau alias, 15 câu hồi quy | 15/15 (100%) | `evaluation/final_legacy_pipeline.json` |
| BM25 mới, 107 câu phát triển | 104/107 (97,2%) | `evaluation/final_dev_bm25.json` |
| BM25 mới, 35 câu giữ riêng | 34/35 (97,1%) | `evaluation/final_test_bm25.json` |

Đối chứng `evaluation/retrieval_ablation.json`: trên 107 câu phát triển, vector top 8 đạt 106/107; pool RRF 20 và 40 đều bao phủ 107/107. Trên 15 câu thông tục, pool 20 bao phủ 14/15, pool 40 bao phủ 15/15. Đây là recall của pool trước rerank, không được so trực tiếp với độ chính xác câu trả lời.

Hai lỗi thông tục được cải thiện: “đuổi việc oan” lấy được Điều 41; câu về trừ lương do đi muộn lấy được Điều 127 về xử lý kỷ luật, thay vì chỉ đưa giới hạn khấu trừ tại Điều 102. Alias không gắn cứng số điều vào câu hỏi.

Bộ ca biên `evaluation/current_edge_pipeline.json`: 12/12 chạy xong không lỗi provider/schema; 6/6 câu ngoài phạm vi/thiếu căn cứ từ chối, 2/2 câu thiếu dữ kiện yêu cầu bổ sung, 4/4 câu có đáp án lấy đúng căn cứ gán nhãn. Bộ nhỏ và nhãn tự soạn, chưa đủ để kết luận tỷ lệ từ chối đúng trên người dùng thực.

Pipeline 15 câu sau alias có p50 khoảng 29,4 giây, p95 khoảng 56,8 giây trên CPU hiện tại. API có quota thấp; báo cáo giữ `previous_errors` khi chạy lại câu lỗi. Độ trễ lần thử thành công không bao gồm toàn bộ thời gian chờ giữa các lần thử.

## Cách đọc và tái lập báo cáo

- Dataset tại `data/evaluation/`; test tách theo điều luật nhưng cùng nguồn câu hỏi lịch sử với dev, không phải tập kiểm tra độc lập bên ngoài.
- JSON giữ câu gốc/rewritten, candidates, context, output, citation, lỗi, thời gian, cấu hình và hash. Benchmark mới bổ sung hash toàn bộ mã Python, kể cả file chưa được Git theo dõi; các báo cáo cũ không có trường này.
- `current_legacy_bm25.json` và `current_dev_bm25.json` là thử nghiệm tokenizer theo âm tiết đã loại bỏ; cấu hình cuối dùng word tokenizer underthesea. `smoke_pipeline.json` ghi lần chạy bị mạng sandbox chặn, không phải điểm chất lượng cuối.
- Chạy các lệnh trong README. Với API quota thấp dùng `--min-interval 60`; `--resume --retry-errors` tiếp tục đúng dataset/corpus/cấu hình. Không chạy nhiều benchmark API cùng lúc.

## Phần chưa được chứng minh

1. Kiểm tra citation hiện chứng minh định danh và đoạn trích có thật, **không chứng minh suy luận được nguồn hỗ trợ đầy đủ**. Các trường `human_review` để trống, không quy đổi sang điểm faithfulness giả định.
2. Nhãn lịch sử chủ yếu cấp điều; chưa có 100–200 mẫu đã được chuyên gia duyệt ở cấp khoản/điểm, đáp án chuẩn và ngoại lệ. Có audit cấu trúc 23 điều, chưa thay thế đối chiếu pháp lý thủ công.
3. Corpus chưa xác minh đầy đủ sửa đổi/hiệu lực hiện hành. Mở rộng luật nằm ngoài phạm vi đợt này theo lựa chọn của người dùng.
4. Ngưỡng raw logit -10 còn tạm thời; chưa có hiệu chỉnh đủ lớn về false acceptance/false refusal. Không diễn giải logit như xác suất.
5. Hợp đồng đã có kiểm thử tách phần, không mất ký tự, truy xuất riêng, OCR và giới hạn xử lý; chưa có bộ hợp đồng có nhãn để đo tỷ lệ phát hiện/báo sai.
6. Chưa xác nhận chạy sạch trên máy mới, Qdrant server tải thật, GPU, Gemini/Ollama và nhiều người dùng đồng thời. Chưa đo đầy đủ token/chi phí; chưa thêm cache câu trả lời khi thiếu bằng chứng cần thiết.

Ưu tiên tiếp theo: duyệt nhãn và câu trả lời đã lưu, lập tập test mới từ lỗi thực tế, rồi mới tối ưu model/ngưỡng hoặc mở rộng dữ liệu. Theo dõi chi tiết tại [ROADMAP](../ROADMAP.md).
