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

        # Fetch nhieu hon tu moi nguon de RRF co pool du rong
        fetch_k = top_k + 15  # e.g. top_k=20 -> fetch 35 moi nguon

        # Tao expanded query de cai thien recall
        expanded_query = self._expand_query(query)

        # Vector search - dung ca query goc lan expanded neu khac nhau
        query_vec = self.embedder.embed_single(query)
        vector_results = self.vector_store.search(
            query_vec, top_k=fetch_k, domain_filter=domain
        )

        if expanded_query and expanded_query != query:
            expanded_vec = self.embedder.embed_single(expanded_query)
            expanded_vector = self.vector_store.search(
                expanded_vec, top_k=fetch_k // 2, domain_filter=domain
            )
            # Merge ket qua expanded vao vector_results
            seen_ids = {self._get_chunk_id(r) for r in vector_results}
            for r in expanded_vector:
                if self._get_chunk_id(r) not in seen_ids:
                    vector_results.append(r)

        # BM25 search - dung expanded query
        bm25_query = expanded_query if (expanded_query and expanded_query != query) else query
        bm25_results = self.bm25.search(bm25_query, top_k=fetch_k)

        # Reciprocal Rank Fusion -> tra ve top_k cho reranker
        return self._rrf_merge(vector_results, bm25_results, top_k)

    def _get_chunk_id(self, item: dict) -> str:
        return item["metadata"].get("chunk_id", item["content"][:50])

    def _expand_query(self, query: str) -> str:
        """Mo rong query de cai thien recall cho cac truong hop dac biet"""
        if not query:
            return query
        q_lower = query.lower()

        # Truy van phu dinh "khong duoc don phuong cham dut"
        # -> them cac truong hop cu the trong Dieu 37
        if "kh\u00f4ng \u0111\u01b0\u1ee3c" in q_lower and (
            "\u0111\u01a1n ph\u01b0\u01a1ng" in q_lower or "ch\u1ea5m d\u1ee9t" in q_lower
        ):
            return (query + " \u1ed1m \u0111au thai s\u1ea3n b\u1ec7nh ngh\u1ec1 nghi\u1ec7p"
                    " tai n\u1ea1n lao \u0111\u1ed9ng c\u1ea5m ch\u1ea5m d\u1ee9t")

        # Hinh thuc ky luat -> them cac hinh thuc cu the (Dieu 124)
        if "h\u00ecnh th\u1ee9c" in q_lower and "k\u1ef7 lu\u1eadt" in q_lower:
            return (query + " khi\u1ec3n tr\u00e1ch c\u1ea3nh c\u00e1o sa th\u1ea3i"
                    " k\u00e9o d\u00e0i n\u00e2ng l\u01b0\u01a1ng")

        # Sa thai khi nao -> them tu khoa noi dung Dieu 125
        if "sa th\u1ea3i" in q_lower and (
            "khi n\u00e0o" in q_lower or "\u00e1p d\u1ee5ng" in q_lower
        ):
            return (query + " tr\u1ed9m c\u1eafp tham \u00f4 \u0111\u00e1nh b\u1ea1c"
                    " ma t\u00fay ti\u1ebft l\u1ed9 b\u00ed m\u1eadt")

        return query

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