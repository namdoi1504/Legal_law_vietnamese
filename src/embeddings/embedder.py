# src/embeddings/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np
from configs.settings import settings


class LegalEmbedder:
    def __init__(self):
        print(f"Loading embedding model: {settings.EMBED_MODEL}")
        self.model = SentenceTransformer(
            settings.EMBED_MODEL,
            device="cpu"  # change to cuda if GPU is available
        )

    def embed_single(self, text: str) -> list[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=settings.EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return vecs.tolist()