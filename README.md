# Legal RAG

A Vietnamese legal assistant for question answering and employment contract review.
The current corpus is a snapshot of Vietnam's Labor Code 2019: 220 articles.

Answers include source references and verified verbatim quotations.
Current amendments and legal validity have not been fully verified.
Use this release for development and review; legal conclusions need human validation.

## Architecture

```text
Source HTML → article/clause/point parser → versioned index snapshot
Question → original query + expansion/rewrite → BM25 + BGE-M3 → RRF
         → BGE reranker → budgeted context → LLM → citation validation
PDF/DOCX → contract sections → independent retrieval → section reports
```

Default retrieval uses 40 candidates and reranks the top 8 chunks.
BM25 and Qdrant share a corpus manifest; previous snapshots support rollback.

## Requirements

- Python 3.11 or later.
- A Groq API key and access to the configured model.
- Internet access for dependency installation, initial model downloads, and Groq.
- CPU is supported; embedding and reranking models are downloaded on first use.

Commands below use PowerShell and run from the repository root.

## 1. Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item examble.env .env }
```

On Linux/macOS, activate with `source .venv/bin/activate`.
Dependencies are pinned in `requirements-core.txt`; the project installs locally.

## 2. Configure

Edit `.env` with your API key and model:

```dotenv
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key
GROQ_MODEL=qwen/qwen3.8-27b
QDRANT_MODE=local
ACTIVE_INDEX_PATH=data/indexes/active.json
MODEL_DEVICE=cpu
```

Existing `.env` values override code defaults. Keep API keys out of version control.
Groq is the evaluated provider; Gemini and Ollama adapters have not been live-tested.

## 3. Build the Index

Skip crawling if `data/raw/labor_law_2019.html` already exists.
Skip building if a valid `data/indexes/active.json` and its artifacts already exist.

```powershell
python -m scripts.crawl
python -m scripts.build_index
```

The build validates chunks, creates BM25 and vector indexes, then activates a manifest.
Artifacts are stored under `data/indexes/<snapshot>/`; embeddings are cached.
Initial embedding can take several minutes on CPU.
Local Qdrant requires no Docker container or separate server.

For cached models, set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before building.

## 4. Run

Activate the virtual environment in both terminals.

```powershell
# Terminal 1: API
uvicorn app.main:app --host 127.0.0.1 --port 8000

# Terminal 2: UI
streamlit run frontend/app.py
```

Open the UI at [localhost:8501](http://localhost:8501).
API documentation is available at [localhost:8000/docs](http://localhost:8000/docs).
Use one API worker with local Qdrant because its storage directory is locked.
CPU responses can take tens of seconds.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process liveness |
| `GET /ready` | Index readiness; does not call the LLM |
| `POST /chat` | Question answering with claims, citations, sources, and status |
| `POST /analyze-document` | Multipart PDF/DOCX review with section-level results |

`/chat` accepts `query` and optional `domain`, `document_version`, `as_of_date`.
Date filtering requires verified validity metadata, which the current snapshot lacks.
Document review accepts up to 3 files, 10 MiB each, and an optional `question`.
Scanned PDFs require OCR first; convert legacy `.doc` files to `.docx`.
Unprocessed sections are reported when processing limits or provider failures occur.

## Tests and Evaluation

Offline regression tests do not call a paid LLM API or download models:

```powershell
python -m unittest discover -s tests -v
python -m src.evaluation.benchmark --mode bm25 --dataset data/evaluation/dev.jsonl --output reports/evaluation/dev_bm25.json
```

End-to-end evaluation calls the configured LLM and consumes API quota:

```powershell
python -m src.evaluation.benchmark --mode pipeline --dataset data/evaluation/test.jsonl --limit 35 --output reports/evaluation/test_pipeline.json --min-interval 60
```

Add `--resume --retry-errors` to continue that report and retry failed cases.
Run API benchmarks sequentially. Reports retain per-case results, errors, and timings.
Retrieval hit rate measures source coverage; it does not measure answer correctness.
Dataset labels and semantic citation support still require expert review.

## Troubleshooting and Maintenance

- Missing index: run `python -m scripts.build_index` before starting the API.
- Qdrant lock error: stop other processes using the same local snapshot.
- HTTP 429: wait for `Retry-After`; avoid concurrent API benchmark runs.
- HTTP 503: check provider credentials/model access and backend logs.
- Unsupported PDF: provide a text-based PDF or run OCR before uploading.

To roll back, stop the API and activate an existing snapshot:

```powershell
python -m scripts.activate_index data/indexes/<snapshot>/manifest.json
```

Restart the API after switching snapshots.
Read [PROGRESS.md](PROGRESS.md) to resume development from the latest checkpoint.
See [ROADMAP.md](ROADMAP.md) for remaining work and acceptance criteria.
See [the implementation report](reports/IMPLEMENTATION_REPORT.md) for measured results.
