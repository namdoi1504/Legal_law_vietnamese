# src/vectordb/qdrant_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PointStruct, Filter, FieldCondition, MatchValue
)
from configs.settings import settings


class LegalVectorStore:
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.collection = settings.COLLECTION_NAME

    def create_collection(self, recreate=False):
        exists = self.client.collection_exists(self.collection)
        if exists and not recreate:
            print(f"Collection {self.collection} already exists")
            return
        if exists and recreate:
            self.client.delete_collection(self.collection)

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=settings.VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="domain",
            field_schema="keyword"
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="law_year",
            field_schema="integer"
        )
        print(f"Created collection: {self.collection}")

    def insert_chunks(self, chunks: list[dict], embeddings: list[list[float]]):
        points = []
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            points.append(PointStruct(
                id=i,
                vector=vec,
                payload=chunk["metadata"] | {
                    "content": chunk["content"],
                    "embed_text": chunk["embed_text"],
                    "chunk_id": chunk["chunk_id"],
                }
            ))

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection,
                points=points[i:i+batch_size]
            )
        print(f"Inserted {len(points)} vectors")

    def search(
    self,
    query_vec: list[float],
    top_k: int = 10,
    domain_filter: str = None
    ) -> list[dict]:
        filter_ = None
        if domain_filter:
            filter_ = Filter(must=[
                FieldCondition(
                    key="domain",
                    match=MatchValue(value=domain_filter)
                )
            ])

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vec,
            limit=top_k,
            query_filter=filter_,
            with_payload=True
        ).points

        return [
            {
                "content": r.payload["content"],
                "metadata": {k: v for k, v in r.payload.items()
                            if k != "content"},
                "score": r.score
            }
            for r in results
        ]