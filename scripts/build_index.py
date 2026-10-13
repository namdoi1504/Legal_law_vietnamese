"""Build an isolated corpus snapshot and atomically publish its manifest."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from configs.settings import ROOT, project_path, settings
from src.corpus import corpus_hash
from src.parser.labor_law_parser import LaborLawParser
from src.parser.validator import validate_chunks
from src.retriever.bm25_retriever import BM25LegalRetriever

SOURCE_URL = "https://thuvienphapluat.vn/van-ban/Lao-dong-Tien-luong/Bo-luat-lao-dong-2019-333670.aspx"


def build(source, activate=True, lexical_only=False):
    parser = LaborLawParser(SOURCE_URL)
    parsed = parser.parse_html(Path(source).read_text(encoding="utf-8"))
    validation = validate_chunks(parsed, range(1, 221))
    chunks = [asdict(c) for c in parsed]
    version = corpus_hash(chunks)
    folder = ROOT / "data/indexes" / f"{version[:12]}-{uuid4().hex[:8]}"
    folder.mkdir(parents=True)
    for c in chunks:
        c["metadata"]["corpus_version"] = version
    chunks_path = folder / "chunks.json"
    chunks_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    bm25 = BM25LegalRetriever()
    bm25.build_index(chunks, version)
    bm25.save_index(folder / "bm25.pkl")
    manifest = {
        "schema_version": 2,
        "corpus_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "source_url": SOURCE_URL,
        "chunks_path": str(chunks_path.relative_to(ROOT)),
        "bm25_path": str((folder / "bm25.pkl").relative_to(ROOT)),
        "qdrant_path": str((folder / "qdrant").relative_to(ROOT)),
        "qdrant_mode": settings.QDRANT_MODE,
        "collection": "legal_" + folder.name.replace("-", "_"),
        "embedding_model": settings.EMBED_MODEL,
        "validation": validation,
        "vector_ready": False,
    }
    if not lexical_only:
        import numpy as np
        from src.embeddings.embedder import LegalEmbedder
        from src.vectordb.qdrant_store import LegalVectorStore

        cache = ROOT / "data/embedding_cache"
        cache.mkdir(parents=True, exist_ok=True)
        cache_key = hashlib.sha256(
            (version + settings.EMBED_MODEL).encode()
        ).hexdigest()
        cache_path = cache / (cache_key + ".npy")
        manifest["embeddings_path"] = str(cache_path.relative_to(ROOT))
        if cache_path.exists():
            embeddings = np.load(cache_path, allow_pickle=False).tolist()
        else:
            embedder = LegalEmbedder()
            embeddings = embedder.embed_batch([c["embed_text"] for c in chunks])
            np.save(cache_path, np.asarray(embeddings, dtype=np.float32))
        if len(embeddings) != len(chunks) or any(
            len(v) != settings.VECTOR_SIZE for v in embeddings
        ):
            raise ValueError(
                "Embedding cache has invalid shape; snapshot not published"
            )
        manifest["embeddings_sha256"] = hashlib.sha256(
            cache_path.read_bytes()
        ).hexdigest()
        store = LegalVectorStore(
            collection=manifest["collection"], path=manifest["qdrant_path"]
        )
        try:
            store.create_collection()
            store.insert_chunks(chunks, embeddings)
            records = []
            offset = None
            while True:
                page, offset = store.client.scroll(
                    store.collection,
                    limit=200,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                records.extend(page)
                if offset is None:
                    break
            if {r.payload["chunk_id"] for r in records} != {
                c["chunk_id"] for c in chunks
            }:
                raise ValueError("Vector/BM25 chunk IDs differ; snapshot not published")
            if any(r.payload.get("corpus_version") != version for r in records):
                raise ValueError("Vector corpus version mismatch")
            manifest["vector_ready"] = True
        finally:
            store.close()
    manifest["chunks_sha256"] = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    manifest["bm25_sha256"] = hashlib.sha256(
        (folder / "bm25.pkl").read_bytes()
    ).hexdigest()
    manifest["bm25_tokenizer"] = bm25.tokenizer
    target = folder / "manifest.json"
    target.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if activate and manifest["vector_ready"]:
        active = project_path(settings.ACTIVE_INDEX_PATH)
        active.parent.mkdir(parents=True, exist_ok=True)
        temporary = active.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        os.replace(temporary, active)
    print(
        json.dumps(
            {
                "manifest": str(target),
                "activated": activate and manifest["vector_ready"],
                **validation,
            },
            ensure_ascii=False,
        )
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(ROOT / "data/raw/labor_law_2019.html"))
    parser.add_argument("--no-activate", action="store_true")
    parser.add_argument("--lexical-only", action="store_true")
    args = parser.parse_args()
    build(args.source, not args.no_activate, args.lexical_only)
