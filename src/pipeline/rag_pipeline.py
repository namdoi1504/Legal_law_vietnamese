# src/pipeline/rag_pipeline.py
import logging
from dataclasses import dataclass, field
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.bge_reranker import BGEReranker
from src.llm.factory import get_llm_client
from src.pipeline.prompts import LEGAL_SYSTEM_PROMPT, build_user_prompt
from src.pipeline.query_rewriter import QueryRewriter
from configs.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    answer: str
    retrieved_chunks: list[dict] = field(default_factory=list)
    reranked_chunks: list[dict] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)


class LegalRAGPipeline:
    def __init__(self):
        print("Loading retriever...")
        self.retriever = HybridRetriever()
        print("Loading reranker...")
        self.reranker = BGEReranker()
        print("Loading LLM...")
        self.llm = get_llm_client()
        self.rewriter = QueryRewriter(self.llm)
        print("Pipeline ready.")

    def run(self, query: str, domain: str = None) -> RAGResult:
        # 1. Query Rewriting — dịch ngôn ngữ thông tục sang pháp lý trước khi retrieve
        retrieval_query = query
        if settings.ENABLE_QUERY_REWRITE:
            retrieval_query, was_rewritten = self.rewriter.rewrite(query)
            if was_rewritten:
                logger.info(f"Query rewritten for retrieval: {retrieval_query!r}")

        # 2. Retrieve (dùng retrieval_query đã rewrite)
        candidates = self.retriever.retrieve(retrieval_query, domain=domain)

        # 3. Rerank (dùng retrieval_query để reranker đánh giá đúng ngữ cảnh)
        final_chunks = self.reranker.rerank(retrieval_query, candidates)

        # 4. Hallucination guard
        # BGE reranker tra ve raw logit (co the am), khong phai probability
        # Chi reject khi score cuc ky thap (< -10) hoac khong co ket qua
        if not final_chunks or final_chunks[0].get("rerank_score", -99) < -10:
            return RAGResult(
                answer="Tôi không tìm thấy thông tin liên quan trong dữ liệu pháp luật hiện có.",
                retrieved_chunks=candidates,
                reranked_chunks=final_chunks,
                citations=[]
            )

        # 5. Build prompt + Generate
        # Dùng query GỐC (không rewrite) để LLM trả lời đúng tone với user
        user_prompt = build_user_prompt(query, final_chunks)
        answer = self.llm.chat(
            system_prompt=LEGAL_SYSTEM_PROMPT,
            user_message=user_prompt,
            temperature=0.1
        )

        # Extract citations
        citations = self._extract_citations(final_chunks)

        return RAGResult(
            answer=answer,
            retrieved_chunks=candidates,
            reranked_chunks=final_chunks,
            citations=citations
        )

    def _extract_citations(self, chunks: list[dict]) -> list[str]:
        citations = []
        for c in chunks:
            meta = c["metadata"]
            cite = (
                f"{meta.get('article', '')} - "
                f"{meta.get('article_title', '')} "
                f"({meta.get('law_name', '')} {meta.get('law_year', '')})"
            )
            citations.append(cite)
        return citations
