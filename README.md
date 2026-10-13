# Legal RAG — tra cứu Bộ luật Lao động 2019

Ứng dụng hỏi đáp và đối chiếu hợp đồng với snapshot Bộ luật Lao động 2019. Câu trả lời được tạo từ nguồn truy xuất; mỗi nhận định phải có mã nguồn và đoạn trích kiểm chứng được.

Phạm vi hiện tại là **220 điều trong snapshot đã thu thập**, không phải toàn bộ pháp luật lao động hiện hành. Hiệu lực sau các lần sửa đổi và các nghị định/thông tư liên quan chưa được xác minh đầy đủ.

## Kiến trúc

```text
HTML nguồn → parser điều/khoản/điểm → snapshot + manifest
                                  ├─ BM25 tiếng Việt
                                  └─ BGE-m3 → Qdrant

Câu hỏi gốc + mở rộng thuật ngữ / rewrite
  → BM25 + vector → RRF (40 ứng viên)
  → BGE reranker (8 chunk, giữ được nhiều khoản cùng điều)
  → mở rộng khoản cha trong ngân sách context
  → LLM → kiểm tra JSON, source ID và trích dẫn nguyên văn
  → câu trả lời / hỏi thêm / báo thiếu căn cứ
```

Parser giữ phần dẫn của khoản, nhận diện điểm `đ)` và tránh nhầm đánh số trong phần trích dẫn sửa đổi luật khác. Index mới được xây trong thư mục riêng; manifest chỉ được kích hoạt sau khi BM25 và Qdrant khớp dữ liệu. Bản cũ được giữ để có thể quay lại.

Rà soát PDF/DOCX truy xuất riêng theo từng phần/điều khoản, giữ vị trí trong văn bản trích xuất và nguồn luật. Báo cáo nêu rõ phần chưa xử lý nếu vượt giới hạn.

## Cài đặt

Tiếp tục công việc sau khi khởi động lại: đọc [PROGRESS.md](PROGRESS.md) trước. File ghi phần đã xong, phần đang chạy và lệnh thực hiện bước kế tiếp.

Yêu cầu Python 3.11 trở lên. Các lệnh dưới đây chạy tại thư mục gốc repository.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Chỉ tạo nếu chưa có .env; giữ nguyên API key đã cấu hình.
if (-not (Test-Path .env)) { Copy-Item examble.env .env }
```

Trên Linux/macOS, kích hoạt bằng `source .venv/bin/activate`.

Cấu hình tối thiểu trong `.env`:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key
GROQ_MODEL=qwen/qwen3.8-27b
QDRANT_MODE=local
MODEL_DEVICE=cpu
```

Groq là cấu hình đã dùng trong đợt kiểm chứng. Có adapter Gemini và Ollama, nhưng chưa chạy đánh giá trực tiếp với hai provider này. Model Groq cũ `llama-3.3-70b-versatile` trả lỗi không khả dụng trong tài khoản lúc kiểm tra ngày 13/10/2026; `GROQ_MODEL` vẫn cấu hình được. Nếu `.env` cũ đã đặt model, cần cập nhật giá trị đó vì nó ghi đè mặc định trong mã.

`requirements-core.txt` chứa dependencies trực tiếp đã đối chiếu với môi trường chạy. `requirements.txt` cài các dependencies này và project local, không còn kéo một revision khác của chính repository từ GitHub.

## Chuẩn bị dữ liệu và index

Nếu đã có `data/raw/labor_law_2019.html`, có thể bỏ qua bước crawl:

```powershell
python -m scripts.crawl
python -m scripts.build_index
```

Lần đầu tải model và tạo embedding có thể mất vài phút, tùy mạng và phần cứng. Có cache embedding cho cùng nội dung/model. Nếu các model đã được cache và muốn chạy offline:

```powershell
$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
python -m scripts.build_index
```

Đầu ra:

- `data/indexes/<snapshot>/chunks.json`, `bm25.pkl`, `qdrant/`, `manifest.json`.
- `data/indexes/active.json`: manifest đang được ứng dụng sử dụng.
- Manifest ghi hash nguồn, corpus, artifact, model và kết quả kiểm tra.

Mặc định Qdrant chạy **embedded local**, không cần Docker/server. Chạy API với **một worker** vì local Qdrant khóa thư mục dữ liệu. Benchmark dùng bản index trong bộ nhớ nên có thể chạy mà không mở thêm khóa của ứng dụng.

Nếu dùng Qdrant server, đặt `QDRANT_MODE=server`, `QDRANT_HOST`, `QDRANT_PORT`, khởi động server rồi xây index mới. Chế độ server có mã hỗ trợ nhưng chưa kiểm thử tải thực trong đợt này.

Lệnh phụ:

```powershell
# Chỉ xem JSON parser, không đổi index đang chạy
python -m scripts.parse

# Xây snapshot BM25 để thử nghiệm, không kích hoạt
python -m scripts.build_index --lexical-only --no-activate

# Quay lại snapshot đã có: dừng API trước nếu dùng local Qdrant
python -m scripts.activate_index data/indexes/<snapshot>/manifest.json
```

Khởi động lại API sau khi kích hoạt snapshot khác. Tiến trình đang chạy giữ index đã nạp lúc khởi động.

## Chạy ứng dụng

```powershell
# Terminal 1
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2
streamlit run frontend/app.py
```

Frontend gọi backend, vì vậy cần chạy cả hai. API docs: [Swagger UI](http://localhost:8000/docs).

| Endpoint | Chức năng |
|---|---|
| `POST /chat` | Hỏi đáp, trả `answer`, `status`, `claims`, `citations`, `sources`, `scope`, `latency_ms` |
| `POST /analyze-document` | PDF có text hoặc DOCX; trả kết quả theo từng phần và số phần chưa xử lý |
| `GET /health` | Kiểm tra tiến trình hoạt động |
| `GET /ready` | Kiểm tra index; không gọi API LLM có tính phí |

`/chat` nhận `query`, tùy chọn `domain`, `document_version`, `as_of_date` (`YYYY-MM-DD`). Filter ngày chỉ chấp nhận dữ liệu đã xác minh hiệu lực; snapshot hiện chưa có nhãn hiệu lực đã xác minh nên yêu cầu lọc ngày sẽ không trả căn cứ.

Các trạng thái: `answered`, `partial`, `needs_information`, `insufficient_evidence`, `validation_failed`. Lỗi provider được API trả HTTP 503; hết quota trả HTTP 429 kèm `Retry-After`. Citation hợp lệ chứng minh đoạn trích tồn tại trong context; **chưa tự chứng minh mọi diễn giải pháp lý đều đúng**.

## Kiểm thử và đánh giá

Kiểm thử hồi quy không gọi API LLM và không tải model:

```powershell
python -m unittest discover -s tests -v
```

Benchmark mặc định chỉ chạy BM25, không dùng API:

```powershell
python -m src.evaluation.benchmark --mode bm25 --dataset data/evaluation/dev.jsonl --output reports/evaluation/dev_bm25.json
python -m src.evaluation.benchmark --mode vector --dataset data/evaluation/dev.jsonl --output reports/evaluation/dev_vector.json
python -m src.evaluation.benchmark --mode hybrid --dataset data/evaluation/dev.jsonl --output reports/evaluation/dev_hybrid.json

# Có gọi API và tiêu thụ quota của provider
python -m src.evaluation.benchmark --mode pipeline --dataset data/evaluation/test.jsonl --limit 35 --output reports/evaluation/test_pipeline.json

# Tiếp tục báo cáo đã có, chỉ chạy lại câu lỗi; tránh chạy API song song
python -m src.evaluation.benchmark --mode pipeline --dataset data/evaluation/test.jsonl --limit 35 --output reports/evaluation/test_pipeline.json --resume --retry-errors --min-interval 60

# So sánh BM25/vector và recall của pool RRF 20/40, không gọi LLM
python -m scripts.retrieval_ablation
```

Dataset gồm 15 câu hồi quy, 107 câu phát triển, 35 câu giữ riêng theo điều luật và 12 ca biên. Nhãn kế thừa chưa được chuyên gia pháp lý kiểm định; xem [hướng dẫn dataset](data/evaluation/README.md). Các câu đã dùng để điều chỉnh không được xem là bằng chứng tổng quát hóa độc lập.

Báo cáo JSON lưu dataset/corpus hash, model/cấu hình, kết quả từng câu, context, câu trả lời, lỗi và độ trễ. Có Hit Rate, Recall, MRR/nDCG; trường rà soát ngữ nghĩa để trống cho đến khi được chấm. Xem [báo cáo triển khai](reports/IMPLEMENTATION_REPORT.md) để biết số đo và hạn chế của bản hiện tại.

Số **96,7% trên 30 câu** trong README cũ là kết quả lịch sử chưa tái lập. Không sử dụng số đó như độ chính xác câu trả lời hiện tại.

## Các giới hạn vận hành

- Reranking bằng BGE trên CPU là bước tốn thời gian nhất; các câu đã đo thường mất hàng chục giây. Điều chỉnh `MODEL_DEVICE` theo phần cứng trước khi tăng concurrency.
- Mặc định: query tối đa 3.000 ký tự; tối đa 3 file, 10 MiB/file, tổng 150.000 ký tự trích xuất và 30 phần/lượt. Phần vượt giới hạn được báo rõ.
- `.doc` cần chuyển sang `.docx`; PDF scan cần OCR trước. DOCX có bảng được đọc theo thứ tự nội dung.
- Giới hạn context sử dụng số byte UTF-8 làm ước lượng bảo thủ để tránh cắt ngang căn cứ; có thể chưa tận dụng hết cửa sổ model.
- Ngưỡng reranker là raw logit, không phải xác suất câu trả lời đúng. Ngưỡng hiện tại còn cần hiệu chỉnh trên nhiều câu ngoài phạm vi hơn.
- Không xác nhận hợp đồng “hoàn toàn hợp pháp” chỉ vì không tìm thấy vi phạm trong corpus hiện có.

Các bước tiếp theo, tiêu chí nghiệm thu và phần còn cần chuyên môn pháp lý được ghi trong [ROADMAP.md](ROADMAP.md).
