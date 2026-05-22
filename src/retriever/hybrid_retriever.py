# src/retriever/hybrid_retriever.py
from src.retriever.bm25_retriever import BM25LegalRetriever
from src.vectordb.qdrant_store import LegalVectorStore
from src.embeddings.embedder import LegalEmbedder
from configs.settings import settings


class HybridRetriever:
    def __init__(self):
        self.embedder = LegalEmbedder()
        self.vector_store = LegalVectorStore()
        self.bm25 = BM25LegalRetriever()
        self.bm25.load_index("data/processed/bm25_index.pkl")

    def retrieve(
            self,
            query: str,
            top_k: int = None,
            domain: str = None
    ) -> list[dict]:
        top_k = top_k or settings.TOP_K_VECTOR

        # Vector search
        query_vec = self.embedder.embed_single(query)
        vector_results = self.vector_store.search(
            query_vec, top_k=top_k, domain_filter=domain
        )

        # BM25 search
        bm25_results = self.bm25.search(query, top_k=top_k)

        # Reciprocal Rank Fusion
        return self._rrf_merge(vector_results, bm25_results, top_k)

    def _rrf_merge(
            self,
            list1: list[dict],
            list2: list[dict],
            top_k: int,
            k: int = 60
    ) -> list[dict]:
        scores = {}

        def get_id(item):
            return item["metadata"].get("chunk_id", item["content"][:50])

        for rank, item in enumerate(list1):
            id_ = get_id(item)
            scores[id_] = scores.get(id_, {"item": item, "score": 0})
            scores[id_]["score"] += 1 / (k + rank + 1)

        for rank, item in enumerate(list2):
            id_ = get_id(item)
            if id_ not in scores:
                scores[id_] = {"item": item, "score": 0}
            scores[id_]["score"] += 1 / (k + rank + 1)

        sorted_results = sorted(
            scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [r["item"] for r in sorted_results[:top_k]]