# api/main.py
import time
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from src.pipeline.rag_pipeline import LegalRAGPipeline
from src.pipeline.document_qa_pipeline import DocumentQAPipeline

app = FastAPI(
    title="Vietnam Legal RAG API",
    description="Hệ thống tra cứu và kiểm tra pháp luật Việt Nam",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = LegalRAGPipeline()
doc_pipeline = DocumentQAPipeline()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_FILES = 3


# Chat
class QueryRequest(BaseModel):
    query: str
    domain: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    latency_ms: float


@app.post("/chat", response_model=QueryResponse)
async def chat(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(400, "Query không được để trống")

    start = time.time()
    result = pipeline.run(req.query, domain=req.domain)
    latency = (time.time() - start) * 1000

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        latency_ms=round(latency, 2)
    )


# Check document upload
@app.post("/analyze-document")
async def analyze_document(
    files: List[UploadFile] = File(...),
    question: Optional[str] = Form(None)
):
    # Validate total file
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Tối đa {MAX_FILES} file")

    # Validate định dạng
    file_data = []
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                400,
                f"File '{f.filename}' không hỗ trợ. Chỉ chấp nhận PDF và Word."
            )
        content = await f.read()
        file_data.append((content, f.filename))

    # run pipeline
    start = time.time()
    result = doc_pipeline.run(file_data, question)
    latency = (time.time() - start) * 1000

    if result.error:
        raise HTTPException(422, result.error)

    return {
        "answer": result.answer,
        "citations": result.citations,
        "files_analyzed": [f.filename for f in files],
        "latency_ms": round(latency, 2)
    }


@app.get("/health")
def health():
    return {"status": "ok"}