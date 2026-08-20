# src/reranker/bge_reranker.py
from sentence_transformers import CrossEncoder
from configs.settings import settings


class BGEReranker:
    def __init__(self):
        # BGE reranker support Vietnamese well
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def rerank(
            self,
            query: str,
            chunks: list[dict],
            top_k: int = None
    ) -> list[dict]:
        top_k = top_k or settings.TOP_K_RERANK

        # Add article metadata into input to reranker 
        pairs = [
            (
                query,
                f"{c['metadata'].get('article', '')} - {c['metadata'].get('article_title', '')}\n{c['content']}"
            )
            for c in chunks
        ]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True
        )

        results = []
        seen_articles = {}
        overflow = []
        for score, chunk in ranked:
            article = chunk["metadata"].get("article", "")
            if article not in seen_articles:
                seen_articles[article] = True
                chunk["rerank_score"] = float(score)
                results.append(chunk)
                if len(results) == top_k:
                    break
            else:
                chunk["rerank_score"] = float(score)
                overflow.append(chunk)

        if len(results) < top_k:
            for chunk in overflow:
                results.append(chunk)
                if len(results) == top_k:
                    break

        return results