import logging
import time
from dataclasses import dataclass, field

from configs.settings import settings
from src.pipeline.answers import validate_answer, render_answer
from src.pipeline.context import build_context, token_upper_bound
from src.pipeline.prompts import LEGAL_SYSTEM_PROMPT, build_user_prompt
from src.llm.base_client import LLMRateLimitError

logger = logging.getLogger(__name__)
SCOPE = "Bộ luật Lao động 2019 trong snapshot đã lập chỉ mục; chưa xác minh đầy đủ sửa đổi và hiệu lực hiện hành."


@dataclass
class RAGResult:
    answer: str
    status: str = "insufficient_evidence"
    retrieved_chunks: list[dict] = field(default_factory=list)
    reranked_chunks: list[dict] = field(default_factory=list)
    context_chunks: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    scope: str = SCOPE
    timings_ms: dict = field(default_factory=dict)
    retrieval_query: str = ""
    error: str | None = None
    retry_after_seconds: float | None = None


class LegalRAGPipeline:
    def __init__(self, retriever=None, reranker=None, llm=None):
        if retriever is None:
            from src.retriever.hybrid_retriever import HybridRetriever

            retriever = HybridRetriever()
        if reranker is None:
            from src.reranker.bge_reranker import BGEReranker

            reranker = BGEReranker()
        if llm is None:
            from src.llm.factory import get_llm_client

            llm = get_llm_client()
        from src.pipeline.query_rewriter import QueryRewriter

        self.retriever, self.reranker, self.llm = retriever, reranker, llm
        self.rewriter = QueryRewriter(llm)

    def retrieve_context(
        self, query, domain=None, document_version=None, as_of_date=None
    ):
        timings = {}
        start = time.perf_counter()
        from src.pipeline.query_rewriter import expand_legal_query, reranking_query
        rewritten = expand_legal_query(query)
        if settings.ENABLE_QUERY_REWRITE:
            rewritten, _ = self.rewriter.rewrite(query)
        timings["rewrite"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        candidates = self.retriever.retrieve(
            query,
            domain=domain,
            rewritten_query=rewritten,
            document_version=document_version,
            as_of_date=as_of_date,
        )
        timings["retrieve"] = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        ranked = self.reranker.rerank(reranking_query(query, rewritten), candidates)
        timings["rerank"] = (time.perf_counter() - start) * 1000
        return candidates, ranked, rewritten, timings

    def run(self, query, domain=None, document_version=None, as_of_date=None):
        query = query.strip()
        if not query or len(query) > settings.MAX_QUERY_CHARS:
            raise ValueError(f"Câu hỏi phải có 1–{settings.MAX_QUERY_CHARS} ký tự.")
        candidates, ranked, rewritten, timings = self.retrieve_context(
            query, domain, document_version, as_of_date
        )
        result = RAGResult(
            answer="Tôi chưa tìm thấy đủ căn cứ trong dữ liệu hiện có để trả lời.",
            retrieved_chunks=candidates,
            reranked_chunks=ranked,
            timings_ms=timings,
            retrieval_query=rewritten,
        )
        if (
            not ranked
            or ranked[0].get("rerank_score", float("-inf"))
            < settings.RERANKER_MIN_SCORE
        ):
            return result
        return self.answer_from_chunks(query, ranked, result)

    def answer_from_chunks(self, query, ranked, result=None):
        result = result or RAGResult(
            answer="Chưa đủ căn cứ để đánh giá.", reranked_chunks=ranked
        )
        if (
            not ranked
            or ranked[0].get("rerank_score", float("-inf"))
            < settings.RERANKER_MIN_SCORE
        ):
            return result
        overhead = (
            token_upper_bound(LEGAL_SYSTEM_PROMPT + build_user_prompt(query, [])) + 256
        )
        budget = min(
            settings.CONTEXT_TOKEN_BUDGET,
            settings.LLM_CONTEXT_TOKENS - settings.LLM_MAX_OUTPUT_TOKENS - overhead,
        )
        context = build_context(ranked, max(0, budget))
        result.context_chunks = context
        if not context:
            result.missing_information = [
                "Không đủ ngân sách context để giữ nguyên căn cứ. Hãy rút gọn câu hỏi."
            ]
            return result
        start = time.perf_counter()
        try:
            raw = self.llm.chat(
                system_prompt=LEGAL_SYSTEM_PROMPT,
                user_message=build_user_prompt(query, context),
                temperature=0,
            )
            payload = validate_answer(raw, context)
            result.answer, result.citations, result.sources = render_answer(
                payload, context
            )
            result.status = payload.status
            result.claims = [c.model_dump() for c in payload.claims]
            result.missing_information = payload.missing_information
        except LLMRateLimitError as exc:
            result.status = 'rate_limited'
            result.error = 'llm_rate_limited'
            result.retry_after_seconds = exc.retry_after
            result.answer = 'Dịch vụ LLM đã chạm hạn mức tạm thời. Vui lòng thử lại sau.'
        except (ValueError, TypeError) as exc:
            logger.warning(
                "Generated answer failed evidence/schema validation: %s",
                type(exc).__name__,
            )
            result.status = "validation_failed"
            result.error = "invalid_generated_evidence"
            result.answer = "Câu trả lời chưa vượt qua kiểm tra trích dẫn. Chưa thể đưa ra kết luận đáng tin cậy."
        except Exception as exc:
            logger.warning("LLM request failed: %s", type(exc).__name__)
            result.status = "provider_error"
            result.error = "llm_unavailable"
            result.answer = (
                "Dịch vụ tạo câu trả lời hiện chưa khả dụng. Vui lòng thử lại."
            )
        result.timings_ms["generate"] = (time.perf_counter() - start) * 1000
        return result
