# Tiến trình Legal RAG — đọc file này trước khi tiếp tục

Checkpoint: 13/10/2026. Thư mục làm việc: `D:\Admin\Python\legal-rag`.

## ĐANG LÀM — bắt đầu từ đây

**Bước hiện tại: hoàn tất benchmark pipeline trên bộ test 35 câu, sau đó chốt kiểm thử và tài liệu.** Phần triển khai chính đã có trong workspace; không cần đọc lại toàn bộ repository hoặc xây lại index.

Tại checkpoint, `reports/evaluation/final_test_pipeline.json` đã lưu **22/35 câu**, câu cuối `formal-103`. Tiến trình benchmark vẫn đang chạy nên JSON có thể đã tiến xa hơn file này. **Đọc `summary` và `cases` trong JSON để biết trạng thái mới nhất.** Mỗi câu hoàn thành được lưu ngay; có thể tiếp tục sau khi đóng ứng dụng/mất tiến trình.

- Có 1 lỗi API tạm thời: `formal-036`, `llm_unavailable` do quota Groq. Cần chạy lại câu lỗi sau khi lượt hiện tại kết thúc.
- Có 1 ca chưa truy xuất đúng nhãn: `formal-024` — “Hình thức xử lý kỷ luật lao động gồm những gì?”. Giữ lỗi này trong báo cáo test; không chỉnh tham số dựa trên test rồi tiếp tục gọi đó là test độc lập.
- Session công cụ hiện tại: `75477`. Đây chỉ là ID trong chat đang mở; lần khởi động sau không được giả định còn dùng được.
- Không chạy hai benchmark API đồng thời. Quota quan sát được khoảng 1.000 output token/phút; chạy song song đã gây lỗi 429.

### Các bước tiếp tục theo thứ tự

1. Kiểm tra benchmark cũ còn chạy hay không. Nếu đang chạy, chờ nó lưu đủ 35 câu. Nếu đã dừng, dùng lệnh resume dưới đây; không chạy lại toàn bộ câu đã thành công.
2. Sau khi đủ 35 câu, dùng `--resume --retry-errors --min-interval 60` để chạy lại lỗi API. Nếu vẫn lỗi, ghi rõ số lỗi; không loại chúng khỏi mẫu số.
3. Bộ unittest cuối đã đạt **44/44**. Chỉ chạy lại nếu sửa mã thêm hoặc phát hiện vấn đề mới.
4. Đã kiểm tra cú pháp 48 file Python và `git diff --check` thành công; Git chỉ cảnh báo chuẩn hóa CRLF/LF. Định dạng các file vừa sửa nếu cần. Không reset thay đổi có sẵn của người dùng.
5. Cập nhật kết quả cuối vào `reports/IMPLEMENTATION_REPORT.md`, `ROADMAP.md` và **file này**. Ghi rõ kết quả truy xuất khác với độ đúng của câu trả lời.
6. Báo lại kết quả thực, số test đạt và giới hạn còn lại. Không tuyên bố hoàn tất toàn bộ yêu cầu pháp lý/production trong roadmap khi chưa có bằng chứng.

### Lệnh dùng ngay (PowerShell tại thư mục project)

Xem số câu đã lưu và lỗi:

```powershell
$r = Get-Content reports/evaluation/final_test_pipeline.json -Raw | ConvertFrom-Json
$r.summary
$r.cases | Where-Object { $_.error } | Select-Object id, error
```

Tiếp tục đánh giá (dùng API đã được người dùng cho phép):

```powershell
$env:PYTHONUTF8='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:GROQ_MODEL='qwen/qwen3.8-27b'
.venv/Scripts/python.exe -B -m src.evaluation.benchmark --mode pipeline --dataset data/evaluation/test.jsonl --limit 35 --output reports/evaluation/final_test_pipeline.json --resume --retry-errors --min-interval 60
```

Kiểm thử không gọi LLM:

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
git diff --check
```

Trong môi trường Codex trên Windows, `TestClient` cần socket nội bộ; sandbox từng làm treo `asyncio.socketpair`. Chạy unittest với quyền thực thi ngoài sandbox nếu gặp tình huống đó. Benchmark cần mạng tới Groq; không in API key hoặc nội dung `.env` ra log.

## ĐÃ XONG

### 1. Phạm vi và roadmap

- Đã đọc/đánh giá kiến trúc và viết `ROADMAP.md`.
- Người dùng đã chốt: **giữ Bộ luật Lao động 2019**, đánh giá API **tối đa 200 câu**. Không hỏi lại quyền cho các đánh giá nằm trong phạm vi này.
- Định hướng: tối giản, đo trước khi thêm thành phần; chưa cần agent workflow, fine-tune hoặc đổi embedding.

### 2. Parser và index

- Giữ cấu trúc điều/khoản/điểm và phần dẫn chung; nhận diện `đ)`, tiêu đề nhiều dòng, bỏ heading chương bị lẫn; xử lý đánh số trong nội dung trích dẫn sửa đổi Điều 219.
- Metadata nguồn/hash/phiên bản/parent; trường hiệu lực chưa xác minh để trống.
- ID chunk ổn định, Qdrant UUID ổn định; kiểm tra độ dài embedding và trùng ID.
- Build BM25/vector trong snapshot riêng, kiểm tra hash/nội dung/phiên bản trước khi kích hoạt; giữ bản cũ và có lệnh rollback.
- Đã xây và kích hoạt snapshot **873 chunk / 220 điều**, không có lỗi cấu trúc.
- Active manifest: `data/indexes/active.json`; snapshot `data/indexes/6aa05971e2a6-86930d2e/manifest.json`; corpus version bắt đầu `6aa05971e2a6`.
- Qdrant embedded local; không cần server. API dùng một worker. Benchmark dựng index bộ nhớ từ embedding cache, không tranh khóa với app.

### 3. Retrieval, context và câu trả lời

- BM25 dùng word tokenizer underthesea; đã bỏ thử nghiệm tokenizer âm tiết vì chất lượng giảm.
- Hợp nhất câu gốc/rewrite bằng RRF; filter hai nguồn; giữ số và phủ định khi rewrite.
- Alias thông tục cho “đuổi việc”, “việc oan”, “trừ lương”, “nghỉ đẻ”; không chèn số điều; câu gốc được giữ.
- 40 candidates, rerank top 8 bằng raw logits; không ép một chunk mỗi điều.
- Mở rộng khoản cha, khử trùng lặp; giữ nguyên căn cứ trong ngân sách context.
- Đầu ra JSON, nhận định gắn đoạn trích; kiểm tra ID/quote và chỉ hiển thị nguồn đã dùng.
- Trạng thái trả lời/thiếu dữ kiện/thiếu căn cứ/lỗi kiểm tra/lỗi provider; xử lý quota riêng HTTP 429 và Retry-After.

### 4. Rà soát tài liệu và API

- PDF/DOCX, đọc bảng DOCX đúng thứ tự; phát hiện scan cần OCR, chấp nhận trang trắng ngăn cách. Bỏ `.doc`.
- Tách điều khoản, giữ offset/ký tự; truy xuất riêng từng phần, kể cả cuối tài liệu; tổng hợp trực tiếp, không thêm bước LLM suy diễn.
- Giới hạn file/query/số phần; báo phần chưa xử lý; dừng khi provider hết quota/lỗi.
- FastAPI lifecycle chia sẻ model, threadpool và semaphore, health/readiness, CORS, lỗi có kiểm soát.
- Frontend nêu phạm vi, trạng thái xử lý một phần và timeout phù hợp hơn.

### 5. Đánh giá, kiểm thử và tài liệu

- Dataset: legacy 15, dev 107, test 35, edge 12 tại `data/evaluation/`.
- Benchmark lưu từng câu, hash dataset/corpus/cấu hình, retrieval/context/answer/citation/lỗi/độ trễ; có resume/retry-errors. Bản mã mới bổ sung hash cây source Python, kể cả file chưa tracked.
- Ablation BM25/vector/RRF pool 20/40 đã chạy.
- **44/44 unittest đạt**, bao gồm test dừng hợp đồng khi hết quota. Cú pháp 48 file Python và `git diff --check` đã đạt. CI không gọi API trả phí.
- README đã viết lại; dependencies trực tiếp đã pin theo môi trường hiện tại; `requirements.txt` không tự cài bản GitHub khác của project.
- Báo cáo triển khai đã tạo: `reports/IMPLEMENTATION_REPORT.md`. Cần bổ sung kết quả test35 cuối cùng.

## KẾT QUẢ ĐÃ CHỐT

| Phép đo | Kết quả | File trong `reports/evaluation/` |
|---|---:|---|
| BM25 cũ, legacy | 10/15 | `baseline_legacy_bm25.json` |
| BM25 mới, legacy | 11/15 | `final_legacy_bm25.json` |
| Pipeline trước alias, legacy | 14/15 | `current_legacy_pipeline.json` |
| Pipeline sau alias, legacy | **15/15**, 0 lỗi cuối | `final_legacy_pipeline.json` |
| BM25 mới, dev | 104/107 | `final_dev_bm25.json` |
| BM25 mới, test | 34/35 | `final_test_bm25.json` |
| Pipeline, edge | 12 câu, 0 lỗi; từ chối đúng 6/6, hỏi bổ sung 2/2, factual lấy đúng nguồn 4/4 | `current_edge_pipeline.json` |
| Pipeline, test | **Đang chạy — đọc JSON mới nhất** | `final_test_pipeline.json` |

Legacy pipeline sau alias: p50 khoảng 29,4 giây, p95 khoảng 56,8 giây trên CPU. **15/15 là hit retrieval, không phải độ chính xác pháp lý 100%.** Các lần lỗi API trước retry được lưu trong `previous_errors`.

Không dùng `current_legacy_bm25.json`/`current_dev_bm25.json` làm cấu hình cuối: đó là thử nghiệm tokenizer đã loại. `smoke_pipeline.json` là lần lỗi mạng sandbox. Số 96,7%/30 câu trong README ban đầu là kết quả lịch sử chưa tái lập.

## CHƯA XONG / CHƯA ĐƯỢC CHỨNG MINH

### Cần chốt ngay trong đợt hiện tại

- [ ] Hoàn thành test35, retry lỗi quota và ghi nhận ca truy xuất hụt.
- [x] Chạy lại unittest sau thay đổi cuối: 44/44; kiểm tra diff/cú pháp đạt. Định dạng bổ sung chỉ khi cần.
- [ ] Đồng bộ số đo và trạng thái roadmap/report/progress.
- [ ] Nếu tiếp tục kiểm chứng hợp đồng thực: tạo fixture có nhãn và chạy đầu cuối; hiện chưa có điểm đo phát hiện/báo sai.

### Cần dữ liệu/kiểm chứng chuyên sâu sau đó

- [ ] Chuyên gia rà soát nhãn và đáp án. Các trường `human_review` còn trống; validator hiện kiểm tra nguồn/quote, chưa kiểm chứng ý nghĩa suy luận.
- [ ] Bộ test mới độc lập, nhãn khoản/điểm/ngoại lệ và đáp án chuẩn. Dev/test hiện tách theo điều nhưng cùng nguồn câu hỏi lịch sử.
- [ ] Đối chiếu pháp lý thủ công ít nhất 20 điều; audit cấu trúc 23 điều đã có nhưng không thay thế việc này.
- [ ] Hiệu chỉnh ngưỡng từ chối trên đủ câu trả lời được/không được; raw logit -10 còn tạm thời.
- [ ] Xác minh sửa đổi/hiệu lực, nguồn chính thức và quan hệ văn bản. Mở rộng corpus chỉ khi phạm vi được thay đổi.
- [ ] Benchmark hợp đồng có nhãn, đo missed issues và false positives.
- [ ] Cài sạch trên máy mới, tải nhiều người dùng, Qdrant server/GPU, thử trực tiếp Gemini/Ollama.
- [ ] Đo token/chi phí đầy đủ; chỉ thêm cache câu trả lời khi có số đo cho thấy cần.

## Bản đồ file để sửa đúng chỗ

| Việc | File chính |
|---|---|
| Benchmark/resume/metrics | `src/evaluation/benchmark.py`, `src/evaluation/metrics.py` |
| Parser/chunk | `src/parser/labor_law_parser.py`, `validator.py` |
| Snapshot/build/rollback | `src/corpus.py`, `scripts/build_index.py`, `scripts/activate_index.py` |
| Retrieval/rerank/rewrite | `src/retriever/`, `src/reranker/`, `src/pipeline/query_rewriter.py` |
| Grounding/context | `src/pipeline/answers.py`, `context.py`, `prompts.py`, `rag_pipeline.py` |
| Rà soát tài liệu | `src/pipeline/document_qa_pipeline.py`, `src/parser/document_reader.py` |
| API/provider | `app/main.py`, `src/llm/groq_client.py`, `base_client.py` |
| Cấu hình/test | `configs/settings.py`, `tests/test_rag.py`, `test_api.py`, `test_snapshot.py` |

## Lưu ý môi trường và thay đổi có sẵn

- Python `.venv/Scripts/python.exe` 3.11.9; BGE-M3 và BGE-reranker-v2-m3 đã cache; CPU, không có GPU. Không rebuild embeddings nếu corpus/model không đổi.
- Groq model chạy được: `qwen/qwen3.8-27b`. Model cũ trả 404 trong tài khoản lúc thử. Không sửa/lộ khóa `.env`; biến môi trường trong lệnh trên chỉ chọn model.
- Workspace chưa commit. Từ đầu đã có `D Chris.py`, thay đổi settings và các `.pyc` tracked; không khôi phục/xóa các thay đổi đó. `.gitignore` mới không làm các `.pyc` đã tracked tự biến mất.
- Black được cài tạm ở `data/devtools` (ignored). CLI Black từng treo do socket sandbox; nếu cần định dạng dùng API `black.format_file_contents` và apply_patch từng file. Không bắt buộc chạy lại toàn repo.
- Không spawn sub-agent nếu người dùng chưa yêu cầu.

**Quy tắc bàn giao:** trước khi kết thúc lần làm tiếp theo, sửa mục “ĐANG LÀM”, số test và danh sách chưa xong trong file này; JSON benchmark là nguồn sự thật cho tiến trình từng câu.
