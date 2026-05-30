# Legal RAG - Hệ thống hỏi đáp pháp luật lao động Việt Nam

Hệ thống RAG (Retrieval-Augmented Generation) tra cứu và hỏi đáp Bộ luật Lao động Việt Nam 2019, đạt **96.7% Hit Rate@8** trên benchmark 30 câu hỏi.

## Kiến trúc

```
Câu hỏi
   ↓
Hybrid Retriever (BM25 + Vector BGE-m3)
   ↓ RRF Fusion (35 candidates mỗi nguồn)
BGE Reranker v2-m3 (top 8, article diversity)
   ↓
LLM (Groq / Gemini / Ollama)
   ↓
Câu trả lời + Trích dẫn điều luật
```

## Tính năng

- **Hỏi đáp pháp luật**: Trả lời câu hỏi về Bộ luật Lao động 2019 với trích dẫn điều luật cụ thể
- **Kiểm tra hợp đồng**: Upload file PDF/DOCX, phát hiện điều khoản vi phạm pháp luật
- **Hybrid Search**: Kết hợp BM25 (từ khóa) + Vector (ngữ nghĩa) + Reranker
- **Query Expansion**: Tự động mở rộng truy vấn phủ định và câu hỏi đặc thù

## Cài đặt

```bash
# 1. Clone repo
git clone https://github.com/ArthurDoi/Legal_law_vietnamese.git
cd legal-rag

# 2. Tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Cài dependencies
pip install -r requirements.txt

# 4. Cấu hình môi trường
cp examble.env .env
# Chỉnh sửa .env: thêm GROQ_API_KEY hoặc GEMINI_API_KEY
```

## Cấu hình `.env`

```env
LLM_PROVIDER=groq          # groq | gemini | ollama
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIza...
```

## Chạy ứng dụng

```bash
# FastAPI backend
uvicorn app.main:app --reload

# Hoặc Streamlit frontend
streamlit run frontend/app.py
```

## Đánh giá độ chính xác

```bash
python -m src.evaluation.benchmark
```

| Retriever | Hit Rate@8 |
|-----------|-----------|
| BM25 | 93.3% |
| Vector (BGE-m3) | 96.7% |
| **Hybrid + Reranker** | **96.7%** |

## Cấu trúc dự án

```
legal-rag/
├── app/                    # FastAPI backend
├── frontend/               # Streamlit UI
├── src/
│   ├── embeddings/         # BGE-m3 embedder
│   ├── retriever/          # BM25 + Vector + Hybrid retriever
│   ├── reranker/           # BGE-reranker-v2-m3
│   ├── pipeline/           # RAG pipeline + Document QA pipeline
│   ├── llm/                # Groq / Gemini / Ollama client
│   └── evaluation/         # Benchmark
├── configs/settings.py     # Cấu hình
├── data/
│   ├── processed/          # JSON chunks + BM25 index
│   └── qdrant_storage/     # Vector database
└── requirements.txt
```

## Model sử dụng

| Thành phần | Model |
|-----------|-------|
| Embedding | `BAAI/bge-m3` |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| LLM (mặc định) | `llama-3.3-70b-versatile` (Groq) |

## Dữ liệu

Bộ luật Lao động Việt Nam 2019 — 220 điều, được chunk và index vào Qdrant vector database.
