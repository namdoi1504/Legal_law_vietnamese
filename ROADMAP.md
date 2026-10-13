# Lộ trình cải thiện Legal RAG

Ngày lập: 13/10/2026. Trạng thái: đã triển khai phần kỹ thuật chính, đang hoàn tất đánh giá. **Checkpoint và bước cần làm ngay: [PROGRESS.md](PROGRESS.md)**. Các checklist chi tiết dưới đây giữ tiêu chí nghiệm thu ban đầu; ô chưa đánh dấu không đồng nghĩa chưa có mã triển khai. Phân biệt phần đã làm và phần chưa được chứng minh trong file tiến trình.

## 1. Mục tiêu và cách sử dụng

Mục tiêu là xây dựng hệ thống hỏi đáp và rà soát hợp đồng lao động có căn cứ kiểm chứng được, bảo toàn điều kiện/ngoại lệ trong văn bản luật và biết khi nào chưa đủ thông tin để kết luận.

Dự án hiện đã có kiến trúc RAG: parser → embedding/Qdrant → BM25 + vector/RRF → reranker → LLM. Lộ trình này nâng chất lượng và khả năng kiểm chứng của hệ thống hiện có.

Cách thực hiện:

1. Làm lần lượt từ giai đoạn 0; mỗi thay đổi chỉ giải quyết một nhóm vấn đề.
2. Trước khi sửa, ghi baseline và tạo ca kiểm chứng cho lỗi cần xử lý.
3. Chỉ đánh dấu `[x]` khi có đầu ra và bằng chứng đáp ứng tiêu chí hoàn thành.
4. Sau mỗi giai đoạn, cập nhật bảng tiến độ cuối tài liệu: thay đổi, kiểm chứng, kết quả và việc còn lại.
5. Dùng tập phát triển để chọn tham số. Chỉ dùng tập kiểm tra độc lập ở các mốc nghiệm thu; nếu dùng kết quả đó để điều chỉnh, cần một tập kiểm tra mới cho lần đánh giá cuối.
6. Khi thay schema/chunking, xây lại cả BM25 và vector index theo cùng phiên bản; giữ bản cũ để có thể quay lại.

Các đường dẫn dưới đây tính từ thư mục gốc repository. File/thư mục ghi “đề xuất” chưa mặc định tồn tại.

## 2. Điểm xuất phát

Các nhận xét sau dựa trên README và mã nguồn đã đọc, chưa phải kết quả benchmark toàn hệ thống:

| Hiện trạng | Ý nghĩa đối với lộ trình |
|---|---|
| README công bố 96,7% Hit Rate@8 trên 30 câu; `BENCHMARK_QUESTIONS` đang hoạt động có 15 câu | Cần đồng bộ dataset và báo cáo trước khi dùng số này làm baseline |
| Benchmark chủ yếu kiểm tra có điều mong đợi trong top 8 | Chưa chứng minh lấy đúng khoản/điểm hoặc trả lời đúng |
| Parser tách các điểm nhưng bỏ phần dẫn chung; regex `[a-z]` không nhận riêng `đ)` | Cần sửa trước khi tối ưu retrieval; đã tái hiện bằng văn bản mẫu |
| Reranker ưu tiên một chunk mỗi điều | Có thể loại khoản liên quan dù có điểm cao khi đã đủ số điều khác nhau |
| `citations` liệt kê context, chưa gắn từng kết luận với bằng chứng | Cần phân biệt nguồn đã truy xuất và nguồn được dùng để trả lời |
| Rà soát hợp đồng truy xuất một lần; không có câu hỏi thì lấy tối đa 400 ký tự đầu của các dòng được chọn | Cần truy xuất riêng theo điều khoản để tăng độ bao phủ |
| Guard dùng ngưỡng logit `< -10` | Cần hiệu chỉnh từ chối trên dữ liệu thực, không xem logit là xác suất đúng |
| Qdrant upsert dùng `id=i`, bắt đầu lại từ 0 mỗi lần ingest | Khi bổ sung văn bản mới có nguy cơ ghi đè; cần ID ổn định và phiên bản dữ liệu |

## 3. Thứ tự và thời gian dự kiến

Ước lượng cho một người phát triển tập trung; việc rà soát nhãn pháp lý và chuẩn bị nguồn có thể kéo dài thêm. Tiêu chí hoàn thành quyết định việc chuyển giai đoạn, không phải số ngày.

| Giai đoạn | Ưu tiên | Ước lượng | Phụ thuộc | Kết quả chính |
|---|---|---|---|---|
| 0. Baseline có thể tái lập | P0 | 1–2 ngày | Không | Dataset và báo cáo ban đầu |
| 1. Dữ liệu và parser | P0 | 3–5 ngày | 0 | Chunk đúng cấu trúc, index đồng bộ |
| 2. Đánh giá đầu cuối | P0 | 3–5 ngày | 0; chốt baseline mới sau 1 | Bộ đo retrieval, answer và citation |
| 3. Retrieval và context | P1 | 3–5 ngày | 1, 2 | Truy xuất đủ căn cứ trong ngân sách token |
| 4. Câu trả lời và trích dẫn | P1 | 3–5 ngày | 2, 3 | Kết luận có nguồn, xử lý thiếu dữ kiện |
| 5. Rà soát hợp đồng | P1 | 4–6 ngày | 3, 4 | Báo cáo theo điều khoản |
| 6. Vận hành và mở rộng | P2 | 3–5 ngày | 4, 5 | API ổn định, quản lý phiên bản luật |

Có thể bắt đầu soạn dataset giai đoạn 2 trong lúc làm giai đoạn 1. Chưa cần đổi embedding/LLM, fine-tune hay thêm agent workflow trước khi có bằng chứng về điểm nghẽn.

## 4. Giai đoạn 0 — Chốt baseline và môi trường

### Công việc

- [ ] Kiểm tra môi trường cài đặt, dependencies và trạng thái Qdrant; ghi lệnh khởi động từ một checkout mới.
- [ ] Cập nhật README với các bước crawl/parse/ingest/build BM25 và cách chạy Qdrant thực tế. `QDRANT_PATH` hiện chưa được dùng trong `LegalVectorStore`, không mô tả nó như cấu hình lưu trữ đang hoạt động.
- [ ] Tách dataset benchmark khỏi mã Python; đề xuất `data/evaluation/dev.jsonl` và `test.jsonl`.
- [ ] Lưu bộ 15 câu hiện tại thành bộ hồi quy; tìm lại dataset/kết quả 30 câu nếu có. Nếu không có bằng chứng, ghi rõ số trong README là kết quả lịch sử chưa tái lập.
- [ ] Chạy baseline BM25, vector, hybrid/rerank và pipeline có rewrite. Phân biệt đánh giá retrieval với đánh giá có sinh câu trả lời.
- [ ] Lưu báo cáo vào `reports/evaluation/` (đề xuất): revision mã, phiên bản/hash dataset và corpus, model, prompt, tham số, ngày chạy, lỗi và độ trễ.

### Tiêu chí hoàn thành

- Có báo cáo đầu tiên, liệt kê rõ số câu và mọi trường hợp thất bại.
- Có thể tái lập phần retrieval với cùng snapshot dữ liệu và cấu hình. Với LLM, ghi biến động giữa các lần chạy thay vì yêu cầu đầu ra giống từng chữ.
- README không gọi Hit Rate@8 là độ chính xác câu trả lời.

File liên quan: `README.md`, `src/evaluation/benchmark.py`, `configs/settings.py`, `scripts/`.

## 5. Giai đoạn 1 — Sửa parser và quản lý dữ liệu

### Công việc

- [ ] Lưu quan hệ điều → khoản → điểm; tách trường `clause_number` và `point_label`, bổ sung `parent_id`.
- [ ] Giữ nguyên phần dẫn/điều kiện chung của khoản khi tách điểm. Phân biệt nội dung nguyên bản với nội dung dùng để embedding.
- [ ] Nhận diện điểm `đ)`, dòng xuống hàng và các dấu phân cách xuất hiện trong nguồn thực tế; tránh nhận nhầm số trong nội dung thành đầu khoản.
- [ ] Tách tiêu đề khỏi thân điều; không để heading chương sau lẫn vào điều trước.
- [ ] Bổ sung `document_id`, `document_version`, `source_url`, `source_hash`, `ingested_at`, `effective_from`, `effective_to` khi có căn cứ. Giá trị chưa xác minh để trống và ghi trạng thái, không suy đoán.
- [ ] Dùng ID Qdrant ổn định, chẳng hạn UUID từ văn bản + phiên bản + điều/khoản/điểm, thay cho số thứ tự `id=i`.
- [ ] Định nghĩa cách ingest lại, xóa chunk không còn thuộc snapshot mới và kiểm tra số embedding khớp số chunk; tránh `zip` âm thầm bỏ dữ liệu dư.
- [ ] Xây BM25 và vector index mới cùng `corpus_version`; kiểm tra ID và số lượng khớp trước khi chuyển cấu hình sang bản mới.
- [ ] Viết kiểm thử hồi quy cho lỗi parser thực và ingest hai văn bản không ghi đè nhau.

### Tiêu chí hoàn thành

- Các ca mất phần dẫn và điểm `đ)` đều được sửa và có kiểm thử.
- Đối chiếu đủ 220 điều với snapshot nguồn nếu corpus tiếp tục là toàn bộ Bộ luật Lao động 2019; không có ID trùng hoặc chunk mồ côi.
- Rà soát thủ công ít nhất 20 điều, ưu tiên điều dài, nhiều điểm, có ngoại lệ hoặc dẫn chiếu.
- Ingest cùng snapshot hai lần không tạo bản sao; bổ sung văn bản khác không làm mất dữ liệu cũ.
- Có manifest của corpus/index và hướng dẫn khôi phục phiên bản trước.

File liên quan: `src/parser/labor_law_parser.py`, `src/parser/validator.py`, `src/vectordb/qdrant_store.py`, `scripts/parse.py`, `scripts/ingest.py`, `scripts/build_bm25.py`.

## 6. Giai đoạn 2 — Đánh giá chất lượng đầu cuối

### Dataset

- [ ] Bắt đầu với 100–200 câu đã rà soát nhãn, sau đó mở rộng theo lỗi thực tế.
- [ ] Bao gồm câu trực tiếp, thông tục/không dấu, số liệu, phủ định, ngoại lệ, cần nhiều căn cứ, thiếu dữ kiện, ngoài corpus và câu hỏi theo thời điểm.
- [ ] Mỗi mẫu có `id`, `question`, `type`, `as_of_date` nếu cần, `answerability`, các căn cứ bắt buộc và đáp án/kết luận tham chiếu.
- [ ] Nhãn căn cứ dùng ID văn bản + phiên bản + điều/khoản/điểm; cho phép nhiều căn cứ cần đồng thời hoặc các nhóm căn cứ thay thế tương đương.
- [ ] Tách dev/test theo nhóm tình huống; các câu diễn đạt lại cùng một tình huống nằm cùng split.
- [ ] Rà soát nhãn với người có hiểu biết pháp lý; kiểm tra câu mơ hồ và nhãn chỉ một điều trước khi dùng làm chuẩn.

### Bộ đo

| Nhóm | Chỉ số | Mục đích |
|---|---|---|
| Retrieval | Hit Rate@k, Recall@k cấp khoản/điểm, MRR/nDCG | Tìm đúng, đủ và xếp hạng đúng căn cứ |
| Context | Độ bao phủ căn cứ bắt buộc trong context cuối | Kiểm tra sau khi mở rộng cha và cắt theo token |
| Answer | Đúng kết luận, đủ điều kiện/ngoại lệ, faithfulness | Kiểm tra câu trả lời thực tế |
| Citation | Đúng định danh, hỗ trợ kết luận, bao phủ nhận định | Tránh nguồn đúng tên nhưng không chứng minh đáp án |
| Abstention | Từ chối đúng và từ chối nhầm | Đo trên cả câu có và không có đủ căn cứ |
| Vận hành | p50/p95 latency, token/chi phí, tỷ lệ lỗi | Đánh giá khả năng sử dụng thực tế |

- [ ] Chạy cả `LegalRAGPipeline.run()` khi đánh giá đầu cuối; benchmark chỉ retrieve/rerank chưa thay thế được bước này.
- [ ] Dùng Ragas hoặc rubric chấm tự xây nếu phù hợp; ghi phiên bản và prompt của LLM chấm. Đối chiếu kết quả chấm với mẫu được con người rà soát.
- [ ] Báo cáo theo từng loại câu và kèm số mẫu/mẫu số, không chỉ điểm trung bình.
- [ ] Lưu query gốc/rewritten, ID/score từng bước, context cuối, answer và citation để phân tích lỗi.

### Tiêu chí hoàn thành

- Chạy một lệnh tạo báo cáo retrieval và một chế độ đánh giá đầu cuối có ngân sách LLM được ghi rõ.
- Có bảng lỗi phân loại: parse, retrieval, thiếu context, rewrite đổi ý, generation, citation hoặc nhãn benchmark.
- Chốt mục tiêu chất lượng trên dev trước khi đánh giá test. Không tự đặt một tỷ lệ chung thành “tiêu chuẩn RAG”.

File liên quan: `src/evaluation/benchmark.py`; đề xuất `tests/`, `data/evaluation/`, `reports/evaluation/`.

## 7. Giai đoạn 3 — Cải thiện retrieval và xây context

### Công việc

- [ ] So sánh query gốc, query rewrite và hợp nhất cả hai; bảo toàn phủ định, số liệu, chủ thể và thời điểm khi rewrite.
- [ ] Đưa các quy tắc expansion hiện tại vào đánh giá đối chứng; giữ quy tắc chỉ khi cải thiện trên nhóm câu liên quan mà không gây suy giảm đáng kể ở nhóm khác.
- [ ] Áp dụng filter domain/phiên bản/thời điểm thống nhất cho BM25 và vector. Hiện `domain` chỉ được chuyển tới vector search.
- [ ] So sánh chọn thuần score với giới hạn mềm số chunk mỗi điều; tránh loại khoản quan trọng chỉ vì trùng điều.
- [ ] Truy xuất chunk nhỏ rồi mở rộng nội dung cha cần thiết: phần dẫn khoản, điểm liên quan, điều được dẫn chiếu khi cần. Không luôn đưa toàn bộ điều dài vào context.
- [ ] Khử trùng lặp và xây context theo ngân sách token của model; dành chỗ cho system prompt, query, tài liệu và output.
- [ ] Thử số candidate, top-k và cấu hình RRF trên dev; theo dõi Recall trước/sau rerank và độ bao phủ sau khi cắt context.
- [ ] Kiểm chứng ca không có candidates; reranker/pipeline cần trả kết quả có kiểm soát.

### Tiêu chí hoàn thành

- Có bảng đối chứng cùng dataset/corpus; giải thích vì sao chọn cấu hình cuối.
- Các ca cần nhiều khoản cùng điều giữ đủ bằng chứng trong context cuối.
- Filter đúng ở cả hai nguồn; context không vượt ngân sách model được chọn.
- Chất lượng đạt mục tiêu đã chốt ở giai đoạn 2; mọi suy giảm so với baseline đều có phân tích và quyết định ghi lại.

File liên quan: `src/retriever/hybrid_retriever.py`, `src/retriever/bm25_retriever.py`, `src/reranker/bge_reranker.py`, `src/pipeline/query_rewriter.py`, `src/pipeline/prompts.py`.

## 8. Giai đoạn 4 — Câu trả lời có căn cứ và biết giới hạn

### Công việc

- [ ] Định nghĩa đầu ra có cấu trúc: `status`, `answer`, `claims`, `source_ids`, `missing_information`, `scope`.
- [ ] Phân biệt ít nhất: trả lời được, thiếu dữ kiện tình huống, thiếu căn cứ trong corpus. Khi đủ căn cứ một phần, trả lời phần đó và nêu phần chưa xác định.
- [ ] Mỗi nhận định pháp lý gắn nguồn cụ thể; kiểm tra source ID thuộc context của lần trả lời.
- [ ] Tách `retrieved_sources` khỏi `citations` thực sự được dùng; khử trùng lặp citation và hiển thị điều/khoản/điểm, phiên bản, URL nguồn.
- [ ] Kiểm tra mức hỗ trợ của nguồn cho kết luận bằng rubric/validator phù hợp. Source ID hợp lệ chưa chứng minh câu trả lời đúng.
- [ ] Hiệu chỉnh guard trên dev có câu ngoài phạm vi và câu khó nhưng trả lời được; thống nhất cấu hình thay cho ngưỡng hardcode `< -10` và setting chưa được dùng.
- [ ] Xử lý output sai schema, nguồn không tồn tại, lỗi provider và retry có giới hạn; sau khi vẫn thất bại trả trạng thái có kiểm soát.
- [ ] Xem tài liệu truy xuất/upload là dữ liệu; kiểm tra ca chứa chỉ dẫn “bỏ qua quy tắc” hoặc yêu cầu bịa nguồn.

### Tiêu chí hoàn thành

- Mọi citation được hiển thị đều ánh xạ về nguồn trong context của request.
- Bộ ca hồi quy không xuất hiện kết luận quan trọng thiếu nguồn hỗ trợ; kết quả trên test phải ghi đầy đủ cả lỗi còn lại.
- Hệ thống hỏi thêm khi thiếu dữ kiện thay vì tự điền giả định; nêu đúng phạm vi corpus và phiên bản đang dùng.
- Đánh giá từ chối có cả false acceptance và false refusal, không chỉ đếm số lần từ chối.

File liên quan: `src/pipeline/rag_pipeline.py`, `src/pipeline/prompts.py`, `app/main.py`, `frontend/app.py`.

## 9. Giai đoạn 5 — Rà soát hợp đồng theo từng điều khoản

### Công việc

- [ ] Tách điều khoản hợp đồng, giữ `clause_id`, nội dung và vị trí nguồn; báo rõ nếu PDF không trích xuất được text hoặc cần OCR.
- [ ] Tạo truy vấn riêng từ từng điều khoản/chủ đề và câu hỏi người dùng; retrieve/rerank căn cứ riêng thay cho một context chung lấy từ phần đầu tài liệu.
- [ ] Kết quả từng điều khoản gồm đoạn gốc, vấn đề, căn cứ, đánh giá, điều kiện chưa biết và đề xuất sửa khi có đủ căn cứ.
- [ ] Dùng nhãn thận trọng: có dấu hiệu không phù hợp, phù hợp trong phạm vi đã kiểm tra, chưa đủ căn cứ; tránh xác nhận cả hợp đồng hợp pháp khi corpus hẹp.
- [ ] Tổng hợp từ kết quả có cấu trúc và giữ nguồn gốc bằng chứng; không để bước tổng hợp tự sinh kết luận mới.
- [ ] Chia theo ranh giới điều khoản và token; kiểm tra nội dung nối các phần không mất ký tự, nhất là fallback cắt cứng trong `_split_doc_into_chunks`.
- [ ] Giới hạn số điều khoản/lượt gọi và token tổng; báo phần chưa được xử lý nếu vượt giới hạn.
- [ ] Tạo bộ hợp đồng mẫu đã ẩn danh: lỗi ở đầu/cuối, nhiều chủ đề, điều khoản dài, thiếu thông tin, PDF scan và nội dung gây nhiễu prompt.

### Tiêu chí hoàn thành

- Đo tỷ lệ phát hiện vấn đề và tỷ lệ báo sai trên bộ mẫu có nhãn.
- Vấn đề nằm ở cuối tài liệu được truy xuất căn cứ riêng và được kiểm chứng.
- Mỗi kết quả trỏ được đến đoạn hợp đồng và nguồn luật; tổng hợp giữ nguyên các liên kết này.
- Không có xác nhận “hợp lệ” chỉ vì không tìm được quy định liên quan.

File liên quan: `src/pipeline/document_qa_pipeline.py`, `src/parser/document_reader.py`, `app/main.py`, `frontend/app.py`.

## 10. Giai đoạn 6 — Vận hành và mở rộng corpus

### Công việc

- [ ] Khởi tạo model qua lifecycle của app và chia sẻ thành phần dùng chung giữa hai pipeline để tránh tải lặp.
- [ ] Xử lý tác vụ đồng bộ nặng trong endpoint `async` bằng worker/threadpool thích hợp; giới hạn concurrency theo RAM/GPU và rate limit provider.
- [ ] Thêm timeout, retry có giới hạn, kiểm tra readiness Qdrant/index/provider cần thiết; phân biệt liveness và readiness.
- [ ] Giới hạn độ dài query, kích thước upload, số file và loại file thực tế; cấu hình CORS phù hợp môi trường triển khai.
- [ ] Log request ID, phiên bản corpus/prompt, độ trễ từng bước, token, lỗi và trạng thái trả lời. Mặc định tránh ghi nguyên hợp đồng/thông tin cá nhân vào log.
- [ ] Bổ sung cache với khóa có corpus/model/prompt version; chỉ thêm khi số đo chứng minh có ích.
- [ ] Thiết lập kiểm thử offline cho parser, filter, context, citation và pipeline với client giả; integration test tách riêng, không gọi API trả phí mặc định trong CI.
- [ ] Chọn dependency tương thích, ghi phiên bản, tài liệu setup và cách khôi phục index.
- [ ] Mở rộng corpus theo nhóm lỗi benchmark: ưu tiên nguồn chính thức và văn bản liên quan đã xác minh, kèm quan hệ sửa đổi/thay thế và hiệu lực theo thời gian.
- [ ] Kiểm tra hiệu lực ở cấp quy định khi văn bản bị sửa một phần; không chỉ gắn một cờ hiệu lực cho toàn văn bản.
- [ ] Chỉ tối ưu GPU, model khác hoặc kiến trúc khác sau khi đo điểm nghẽn chất lượng/độ trễ.

### Tiêu chí hoàn thành

- Cài và chạy được từ môi trường sạch theo README; có kiểm thử cho các ca hồi quy quan trọng.
- Đo tải trên phần cứng dự kiến, đáp ứng ngân sách độ trễ/chi phí đã ghi trước thử nghiệm.
- Lỗi Qdrant/provider/upload không làm request treo vô hạn hoặc trả kết luận không có căn cứ.
- Có quy trình xuất bản snapshot dữ liệu mới, kiểm chứng BM25/vector đồng bộ và quay lại snapshot trước.

## 11. Checklist nghiệm thu một phiên bản

- [ ] Parser và ingest vượt qua các ca hồi quy; corpus có nguồn gốc và manifest.
- [ ] Đánh giá dùng dataset đã rà soát, có split độc lập và phiên bản rõ ràng.
- [ ] Báo cáo phân biệt retrieval tốt với câu trả lời đúng; có kết quả theo loại câu và danh sách lỗi.
- [ ] Đáp án đủ điều kiện/ngoại lệ trong phạm vi kiểm tra, trích dẫn có nguồn hỗ trợ.
- [ ] Câu thiếu thông tin/ngoài corpus được xử lý có kiểm soát.
- [ ] Rà soát hợp đồng giữ liên kết điều khoản → kết luận → nguồn luật.
- [ ] Không có suy giảm chưa giải thích ở nhóm lỗi ưu tiên.
- [ ] README phản ánh đúng phạm vi, dữ liệu, cấu hình và kết quả đã đo của phiên bản này.

Mục tiêu số cần chốt sau baseline: Recall@k, tỷ lệ đáp án đúng/đủ, citation support, tỷ lệ từ chối đúng/nhầm, p95 latency và chi phí/request. Đây là mục tiêu nội bộ của dự án, không phải chứng nhận hay ngưỡng RAG phổ quát.

## 12. Bảng theo dõi thực hiện

| Giai đoạn | Trạng thái | Bằng chứng / báo cáo | Kết quả và việc còn lại |
|---|---|---|---|
| 0. Baseline | Đã có baseline và tài liệu | `reports/evaluation/`, README | Chưa kiểm chứng cài sạch trên máy mới |
| 1. Dữ liệu/parser | Đã triển khai và build | Active manifest, parser tests | 873 chunk/220 điều; còn đối chiếu pháp lý thủ công |
| 2. Evaluation | Đang chốt test35 | `PROGRESS.md`, benchmark JSON | Dataset/runner đã có; nhãn và semantic review chưa được chuyên gia duyệt |
| 3. Retrieval/context | Đã triển khai, có ablation | `retrieval_ablation.json`, legacy pipeline | Legacy hit 15/15; còn hiệu chỉnh guard trên tập lớn |
| 4. Answer/citation | Đã triển khai validator | Answer/API tests, edge pipeline | Kiểm tra ID/quote; chưa chứng minh semantic support đầy đủ |
| 5. Hợp đồng | Đã triển khai truy xuất từng phần | Document/reader tests | Chưa có benchmark hợp đồng có nhãn |
| 6. Vận hành/corpus | Đã triển khai vận hành local | API tests, README, CI | Chưa kiểm thử tải/server/provider khác; chưa mở rộng corpus theo phạm vi đã chốt |

Mẫu ghi sau mỗi thay đổi:

```text
Ngày / revision:
Hạng mục roadmap:
Vấn đề và thay đổi:
Dataset / corpus / model / prompt version:
Cách kiểm chứng:
Kết quả so với baseline:
Lỗi hoặc hạn chế còn lại:
Bước tiếp theo:
```

## 13. Tài liệu tham khảo

- [Microsoft — RAG evaluators](https://learn.microsoft.com/en-us/azure/foundry/concepts/evaluation-evaluators/rag-evaluators): phân biệt đánh giá retrieval với groundedness, relevance và completeness của câu trả lời.
- [Ragas — Available metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/): tham khảo các metric hỗ trợ đánh giá RAG. Cần chọn metric phù hợp và đối chiếu với nhãn được con người rà soát.

Việc cần làm tiếp: đọc phần **ĐANG LÀM** trong [PROGRESS.md](PROGRESS.md), tiếp tục benchmark đang dở và chốt kiểm thử/tài liệu; không chạy lại giai đoạn 0 hoặc rebuild index khi chưa có thay đổi dữ liệu.
