from configs.settings import settings, project_path
from src.corpus import load_manifest
from src.retriever.bm25_retriever import BM25LegalRetriever


class HybridRetriever:
    def __init__(self, embedder=None, vector_store=None, bm25=None, manifest=None):
        if manifest and manifest["embedding_model"] != settings.EMBED_MODEL:
            raise ValueError("Embedding model differs from index manifest")
        if bm25 is None:
            manifest = manifest or load_manifest()
            bm25 = BM25LegalRetriever()
            bm25.load_index(project_path(manifest["bm25_path"]))
            if bm25.corpus_version != manifest["corpus_version"]:
                raise ValueError("BM25/manifest corpus mismatch")
        if embedder is None:
            from src.embeddings.embedder import LegalEmbedder

            embedder = LegalEmbedder()
        if vector_store is None:
            from src.vectordb.qdrant_store import LegalVectorStore

            manifest = manifest or load_manifest()
            if manifest["embedding_model"] != settings.EMBED_MODEL:
                raise ValueError("Embedding model differs from index manifest")
            vector_store = LegalVectorStore(
                collection=manifest["collection"],
                path=manifest["qdrant_path"],
                mode=manifest["qdrant_mode"],
            )
            try:
                vector_store.validate_snapshot(bm25.chunks, manifest["corpus_version"])
            except Exception:
                vector_store.close()
                raise
        self.embedder, self.vector_store, self.bm25 = embedder, vector_store, bm25
        self.manifest = manifest or {}

    def retrieve(
        self,
        query,
        top_k=None,
        domain=None,
        rewritten_query=None,
        document_version=None,
        as_of_date=None,
    ):
        if not query.strip():
            return []
        top_k = settings.RETRIEVAL_CANDIDATES if top_k is None else top_k
        if top_k <= 0:
            return []
        if not rewritten_query or rewritten_query == query:
            from src.pipeline.query_rewriter import expand_legal_query
            rewritten_query = expand_legal_query(query)
        queries = list(
            dict.fromkeys(q for q in [query, rewritten_query] if q and q.strip())
        )
        vectors, keywords = [], []
        for q in queries:
            vectors.append(
                self.vector_store.search(
                    self.embedder.embed_single(q),
                    top_k=top_k,
                    domain_filter=domain,
                    document_version=document_version,
                    as_of_date=as_of_date,
                )
            )
            keywords.append(
                self.bm25.search(
                    q,
                    top_k=top_k,
                    domain=domain,
                    document_version=document_version,
                    as_of_date=as_of_date,
                )
            )
        # Merge query variants within each source first so rewriting cannot double its weight.
        return self._rrf_merge(
            self._fuse(vectors, top_k), self._fuse(keywords, top_k), top_k
        )

    @staticmethod
    def _get_chunk_id(item):
        return item["metadata"]["chunk_id"]

    @classmethod
    def _fuse(cls, lists, top_k, k=None):
        scores, items = {}, {}
        for rows in lists:
            seen = set()
            for rank, item in enumerate(rows, 1):
                key = cls._get_chunk_id(item)
                if key in seen:
                    continue
                seen.add(key)
                items[key] = item
                scores[key] = scores.get(key, 0) + 1 / ((k or settings.RRF_K) + rank)
        keys = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
        return [{**items[key], "fusion_score": scores[key]} for key in keys]

    def _rrf_merge(self, list1, list2, top_k, k=None):
        return self._fuse([list1, list2], top_k, k)
