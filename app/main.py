import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.pipeline.rag_pipeline import LegalRAGPipeline

app = FastAPI(
    title="Vietnam Legal RAG API",
    description="Hệ thống tra cứu pháp luật Việt Nam",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi tạo pipeline một lần duy nhất khi server start
pipeline = LegalRAGPipeline()


class QueryRequest(BaseModel):
    query: str
    domain: str = None  # None = tìm tất cả domain


class ChunkInfo(BaseModel):
    article: str
    article_title: str
    content: str
    rerank_score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    chunks_used: list[ChunkInfo]
    latency_ms: float


@app.post("/chat", response_model=QueryResponse)
async def chat(req: QueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query không được để trống")

    start = time.time()

    result = pipeline.run(req.query, domain=req.domain)

    latency = (time.time() - start) * 1000

    chunks_info = [
        ChunkInfo(
            article=c["metadata"].get("article", ""),
            article_title=c["metadata"].get("article_title", ""),
            content=c["content"],
            rerank_score=c.get("rerank_score", 0.0)
        )
        for c in result.reranked_chunks
    ]

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        chunks_used=chunks_info,
        latency_ms=round(latency, 2)
    )


@app.get("/health")
def health():
    return {"status": "ok", "model": "llama-3.1-8b-instant"}


@app.get("/domains")
def list_domains():
    """Liệt kê domain đang có — dùng cho frontend dropdown."""
    return {
        "domains": ["labor", "civil", "criminal", "traffic"],
        "active": ["labor"]  # Cập nhật thủ công khi thêm domain mới
    }