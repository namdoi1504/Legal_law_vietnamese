# src/pipeline/rag_pipeline.py
from dataclasses import dataclass, field
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.bge_reranker import BGEReranker
from src.llm.factory import get_llm_client
from src.pipeline.prompts import LEGAL_SYSTEM_PROMPT, build_user_prompt


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
        print("Pipeline ready.")

    def run(self, query: str, domain: str = None) -> RAGResult:
        # 1. Retrieve
        candidates = self.retriever.retrieve(query, domain=domain)

        # 2. Rerank
        final_chunks = self.reranker.rerank(query, candidates)

        # 3. Hallucination guard
        # BGE reranker tra ve raw logit (co the am), khong phai probability
        # Chi reject khi score cuc ky thap (< -10) hoac khong co ket qua
        if not final_chunks or final_chunks[0].get("rerank_score", -99) < -10:
            return RAGResult(
                answer="Toi khong tim thay thong tin lien quan trong du lieu phap luat hien co.",
                retrieved_chunks=candidates,
                reranked_chunks=final_chunks,
                citations=[]
            )

        # 4. Build prompt + Generate
        user_prompt = build_user_prompt(query, final_chunks)
        answer = self.llm.chat(
            system_prompt=LEGAL_SYSTEM_PROMPT,
            user_message=user_prompt,
            temperature=0.1
        )

        # 5. Extract citations
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
