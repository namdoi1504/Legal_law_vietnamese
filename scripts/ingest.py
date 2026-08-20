# scripts/ingest.py
import json
import sys
import os
from src.embeddings.embedder import LegalEmbedder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectordb.qdrant_store import LegalVectorStore


def run_ingest(parsed_json_path: str):
    # Load parsed data
    with open(parsed_json_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loaded {len(chunks)} chunks")

    embedder = LegalEmbedder()
    texts = [c["embed_text"] for c in chunks]
    embeddings = embedder.embed_batch(texts)

    # Insert into Qdrant
    store = LegalVectorStore()
    store.create_collection(recreate=False)
    store.insert_chunks(chunks, embeddings)

    print("Ingest complete.")


if __name__ == "__main__":
    run_ingest("data/processed/labor_law_2019.json")