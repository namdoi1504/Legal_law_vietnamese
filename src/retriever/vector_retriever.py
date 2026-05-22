from src.vectordb.qdrant_store import LegalVectorStore
from src.embeddings.embedder import LegalEmbedder
from configs.settings import settings


class VectorRetriever:
    def __init__(self):
        self.embedder = LegalEmbedder()
        self.vector_store = LegalVectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        domain: str = None
    ) -> list[dict]:
        top_k = top_k or settings.TOP_K_VECTOR

        query_vec = self.embedder.embed_single(query)
        results = self.vector_store.search(
            query_vec=query_vec,
            top_k=top_k,
            domain_filter=domain
        )
        return results