"""Compare lexical/vector/fusion candidate recall without paid LLM calls."""

import hashlib
import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient
from configs.settings import project_path
from src.corpus import load_manifest
from src.embeddings.embedder import LegalEmbedder
from src.retriever.bm25_retriever import BM25LegalRetriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.vectordb.qdrant_store import LegalVectorStore
from src.evaluation.metrics import retrieval_metrics


def main():
    manifest = load_manifest()
    bm = BM25LegalRetriever()
    bm.load_index(project_path(manifest["bm25_path"]))
    key = hashlib.sha256(
        (manifest["corpus_version"] + manifest["embedding_model"]).encode()
    ).hexdigest()
    vectors = np.load(
        project_path(
            manifest.get("embeddings_path", f"data/embedding_cache/{key}.npy")
        ),
        allow_pickle=False,
    )
    store = LegalVectorStore(client=QdrantClient(":memory:"), collection="ablation")
    store.create_collection()
    store.insert_chunks(bm.chunks, vectors.tolist())
    embedder = LegalEmbedder()
    report = {"corpus_version": manifest["corpus_version"], "cases": []}
    try:
        for split in ("legacy", "dev"):
            items = [
                json.loads(l)
                for l in project_path(f"data/evaluation/{split}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            for item in items:
                vector = store.search(embedder.embed_single(item["question"]), top_k=40)
                lexical = bm.search(item["question"], top_k=40)
                variants = {
                    "bm25@8": lexical[:8],
                    "vector@8": vector[:8],
                    "fusion@20": HybridRetriever._fuse([vector, lexical], 20),
                    "fusion@40": HybridRetriever._fuse([vector, lexical], 40),
                }
                report["cases"].append(
                    {
                        "id": item["id"],
                        "split": split,
                        "metrics": {
                            k: retrieval_metrics(item, v) for k, v in variants.items()
                        },
                    }
                )
            report[split] = {
                k: sum(
                    c["metrics"][k]["hit"]
                    for c in report["cases"]
                    if c["split"] == split
                )
                / len(items)
                for k in variants
            }
            print(split, report[split], flush=True)
    finally:
        store.close()
    project_path("reports/evaluation/retrieval_ablation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
