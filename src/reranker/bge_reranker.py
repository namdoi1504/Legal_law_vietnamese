# src/reranker/bge_reranker.py
from sentence_transformers import CrossEncoder
from configs.settings import settings


class BGEReranker:
    def __init__(self):
        # BGE reranker hỗ trợ tiếng Việt tốt
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def rerank(
            self,
            query: str,
            chunks: list[dict],
            top_k: int = None
    ) -> list[dict]:
        top_k = top_k or settings.TOP_K_RERANK

        pairs = [(query, c["content"]) for c in chunks]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )

        results = []
        for score, chunk in ranked[:top_k]:
            chunk["rerank_score"] = float(score)
            results.append(chunk)

        return results