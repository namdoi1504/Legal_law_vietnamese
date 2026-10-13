import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
import logging
import math
from pathlib import Path
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from configs.settings import settings

logger = logging.getLogger(__name__)


def create_app(pipeline=None, document_pipeline=None):
    @asynccontextmanager
    async def lifespan(app):
        nonlocal pipeline, document_pipeline
        if pipeline is None:
            from src.pipeline.rag_pipeline import LegalRAGPipeline

            pipeline = await run_in_threadpool(LegalRAGPipeline)
        if document_pipeline is None:
            from src.pipeline.document_qa_pipeline import DocumentQAPipeline

            document_pipeline = DocumentQAPipeline(pipeline)
        app.state.pipeline = pipeline
        app.state.document_pipeline = document_pipeline
        app.state.gate = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
        try:
            yield
        finally:
            store = getattr(getattr(pipeline, "retriever", None), "vector_store", None)
            if store:
                store.close()

    app = FastAPI(title="Vietnam Legal RAG API", version="0.2.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        retriever = request.app.state.pipeline.retriever
        try:
            count = retriever.vector_store.client.count(
                retriever.vector_store.collection, exact=True
            ).count
            if count != len(retriever.bm25.chunks):
                raise ValueError("Index count mismatch")
        except Exception as exc:
            raise HTTPException(503, "Index chưa sẵn sàng.") from exc
        return {
            "status": "ready",
            "corpus_version": retriever.manifest.get("corpus_version"),
            "provider_check": "not_probed",
            "chunks": count,
        }

    async def execute(request, fn, *args, **kwargs):
        start = time.perf_counter()
        request_id = uuid4().hex
        try:
            await asyncio.wait_for(request.app.state.gate.acquire(), timeout=2)
        except TimeoutError as exc:
            raise HTTPException(503, "Hệ thống đang bận. Vui lòng thử lại.") from exc
        try:
            result = await run_in_threadpool(fn, *args, **kwargs)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.error("Request %s failed: %s", request_id, type(exc).__name__)
            raise HTTPException(503, "Dịch vụ truy xuất chưa sẵn sàng.") from exc
        finally:
            request.app.state.gate.release()
        latency = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request=%s status=%s latency_ms=%s", request_id, result.status, latency
        )
        return result, request_id, latency

    @app.post("/chat")
    async def chat(req: QueryRequest, request: Request):
        result, request_id, latency = await execute(
            request,
            request.app.state.pipeline.run,
            req.query,
            domain=req.domain,
            document_version=req.document_version,
            as_of_date=req.as_of_date.isoformat() if req.as_of_date else None,
        )
        if result.status == "provider_error":
            raise HTTPException(503, result.answer)
        if result.status == 'rate_limited':
            raise HTTPException(429, result.answer, headers={'Retry-After':str(math.ceil(result.retry_after_seconds or 30))})
        data = asdict(result)
        for key in (
            "retrieved_chunks",
            "reranked_chunks",
            "context_chunks",
            "retrieval_query",
        ):
            data.pop(key, None)
        return {**data, "request_id": request_id, "latency_ms": latency}

    @app.post("/analyze-document")
    async def analyze_document(
        request: Request,
        files: list[UploadFile] = File(...),
        question: str | None = Form(None),
    ):
        if not 1 <= len(files) <= 3:
            raise HTTPException(400, "Cần từ 1 đến 3 file.")
        data = []
        for file in files:
            name = file.filename or ""
            if Path(name).suffix.lower() not in (".pdf", ".docx"):
                raise HTTPException(400, "Chỉ hỗ trợ PDF và DOCX.")
            content = await file.read(settings.MAX_UPLOAD_BYTES + 1)
            if len(content) > settings.MAX_UPLOAD_BYTES:
                raise HTTPException(413, "File vượt giới hạn kích thước.")
            data.append((content, name))
        result, request_id, latency = await execute(
            request, request.app.state.document_pipeline.run, data, question
        )
        if result.error:
            raise HTTPException(422, result.error)
        output = asdict(result)
        output.pop("retrieved_chunks", None)
        return {
            **output,
            "request_id": request_id,
            "latency_ms": latency,
            "files_analyzed": [name for _, name in data],
        }

    return app


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=settings.MAX_QUERY_CHARS)
    domain: str | None = None
    document_version: str | None = None
    as_of_date: date | None = None


app = create_app()
