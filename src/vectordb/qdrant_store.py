from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from configs.settings import project_path, settings
from src.corpus import matches_filters


class LegalVectorStore:
    def __init__(self, client=None, collection=None, path=None, mode=None):
        mode = mode or settings.QDRANT_MODE
        self.client = (
            client
            if client is not None
            else (
                QdrantClient(
                    path=str(project_path(path or settings.QDRANT_PATH)),
                    force_disable_check_same_thread=True,
                )
                if mode == "local"
                else QdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    timeout=settings.QDRANT_TIMEOUT,
                )
            )
        )
        self.collection = collection or settings.COLLECTION_NAME

    def create_collection(self, recreate=False):
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            return
        if exists:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=VectorParams(
                size=settings.VECTOR_SIZE, distance=Distance.COSINE
            ),
        )

    def insert_chunks(self, chunks, embeddings):
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk/embedding count mismatch")
        ids = [c["chunk_id"] for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk IDs")
        points = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, c["chunk_id"])),
                vector=list(vec),
                payload={
                    **c["metadata"],
                    "content": c["content"],
                    "chunk_id": c["chunk_id"],
                },
            )
            for c, vec in zip(chunks, embeddings)
        ]
        for start in range(0, len(points), 100):
            self.client.upsert(
                self.collection, points=points[start : start + 100], wait=True
            )

    def search(
        self,
        query_vec,
        top_k=10,
        domain_filter=None,
        document_version=None,
        as_of_date=None,
    ):
        if top_k <= 0:
            return []
        conditions = []
        for key, value in [
            ("domain", domain_filter),
            ("document_version", document_version),
        ]:
            if value:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        if as_of_date:
            conditions.append(
                FieldCondition(key="validity_verified", match=MatchValue(value=True))
            )
        # Date filtering must precede top-k selection; fetch all eligible metadata when requested.
        limit = (
            self.client.count(self.collection, exact=True).count
            if as_of_date
            else top_k
        )
        if not limit:
            return []
        rows = self.client.query_points(
            self.collection,
            query=query_vec,
            limit=limit,
            query_filter=Filter(must=conditions) if conditions else None,
            with_payload=True,
        ).points
        return [
            {
                "content": r.payload["content"],
                "metadata": {k: v for k, v in r.payload.items() if k != "content"},
                "score": float(r.score),
            }
            for r in rows
            if matches_filters(r.payload, domain_filter, document_version, as_of_date)
        ][:top_k]

    def close(self):
        self.client.close()

    def validate_snapshot(self, chunks, corpus_version):
        expected = {c["chunk_id"]: c["content"] for c in chunks}
        actual = {}
        offset = None
        while True:
            page, offset = self.client.scroll(
                self.collection,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for record in page:
                payload = record.payload
                if payload.get("corpus_version") != corpus_version:
                    raise ValueError("Vector corpus version mismatch")
                key = payload["chunk_id"]
                if key in actual:
                    raise ValueError("Duplicate vector chunk ID")
                actual[key] = payload["content"]
            if offset is None:
                break
        if actual != expected:
            raise ValueError("Vector/BM25 corpus content mismatch")
