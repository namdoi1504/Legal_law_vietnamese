# src/retriever/bm25_retriever.py
import json
import pickle
from rank_bm25 import BM25Okapi
import underthesea


class BM25LegalRetriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build_index(self, chunks: list[dict]):
        self.chunks = chunks
        # Tokenize Vietnamese
        tokenized = [
            underthesea.word_tokenize(c["embed_text"].lower())
            for c in chunks
        ]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built with {len(chunks)} docs")

    def save_index(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunks": self.chunks}, f)

    def load_index(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        tokenized_query = underthesea.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [
            {
                "content": self.chunks[i]["content"],
                "metadata": self.chunks[i]["metadata"],
                "score": float(scores[i])
            }
            for i in top_indices if scores[i] > 0
        ]