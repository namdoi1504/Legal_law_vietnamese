# src/reranker/bge_reranker.py
from sentence_transformers import CrossEncoder
from configs.settings import settings


class BGEReranker:
    def __init__(self):
        # BGE reranker ho tro tieng Viet tot
        self.model = CrossEncoder("BAAI/bge-reranker-v2-m3")

    def rerank(
            self,
            query: str,
            chunks: list[dict],
            top_k: int = None
    ) -> list[dict]:
        top_k = top_k or settings.TOP_K_RERANK

        # Them article metadata vao input de reranker biet ngu canh dieu luat
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
        # Phase 1: Article diversity — lay chunk tot nhat cua moi dieu luat
        # Ngan mot dieu luat chiem nhieu slots, day dieu luat khac ra ngoai top_k
        seen_articles = {}
        overflow = []  # Chunk bi reject vi cung dieu luat da co dai dien
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

        # Phase 2: Neu chua du top_k thi fill them tu overflow
        if len(results) < top_k:
            for chunk in overflow:
                results.append(chunk)
                if len(results) == top_k:
                    break

        return results