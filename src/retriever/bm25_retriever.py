import pickle
import re
import unicodedata
from pathlib import Path

from rank_bm25 import BM25Okapi
from src.corpus import matches_filters


def tokenize(text):
    from underthesea import word_tokenize

    return word_tokenize(text.lower())


class BM25LegalRetriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.corpus_version = None
        self.tokenizer = "underthesea-v1"

    def build_index(self, chunks, corpus_version=None):
        if not chunks:
            raise ValueError("Cannot index an empty corpus")
        self.chunks = chunks
        self.corpus_version = corpus_version
        self.bm25 = BM25Okapi([tokenize(c["embed_text"]) for c in chunks])

    def save_index(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "chunks": self.chunks,
                    "corpus_version": self.corpus_version,
                    "tokenizer": self.tokenizer,
                },
                f,
            )

    def load_index(self, path):
        # Only load trusted local indexes created by this application.
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25, self.chunks = data["bm25"], data["chunks"]
        self.corpus_version = data.get("corpus_version")
        self.tokenizer = data.get("tokenizer", "underthesea-legacy")

    def search(
        self, query, top_k=10, domain=None, document_version=None, as_of_date=None
    ):
        if not query.strip() or top_k <= 0:
            return []
        if self.tokenizer.startswith("underthesea"):
            from underthesea import word_tokenize

            tokens = word_tokenize(query.lower())
        else:
            # Compatibility with the recorded syllable-tokenizer experiment.
            normalized = unicodedata.normalize("NFD", query.lower().replace("đ", "d"))
            tokens = re.findall(
                r"\w+",
                "".join(c for c in normalized if unicodedata.category(c) != "Mn"),
            )
        scores = self.bm25.get_scores(tokens)
        indices = [
            i
            for i, c in enumerate(self.chunks)
            if scores[i] > 0
            and matches_filters(c["metadata"], domain, document_version, as_of_date)
        ]
        indices.sort(key=lambda i: scores[i], reverse=True)
        return [
            {
                "content": self.chunks[i]["content"],
                "metadata": dict(self.chunks[i]["metadata"]),
                "score": float(scores[i]),
            }
            for i in indices[:top_k]
        ]
