"""Shared index manifest and metadata filtering for both retrieval branches."""

import hashlib
import json
from datetime import date

from configs.settings import project_path, settings


def corpus_hash(chunks):
    stable = [{k: c[k] for k in ("chunk_id", "content", "embed_text")} for c in chunks]
    return hashlib.sha256(
        json.dumps(stable, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def load_manifest(path=None, require_vectors=True):
    target = project_path(path or settings.ACTIVE_INDEX_PATH)
    if not target.exists():
        raise FileNotFoundError(
            "Chưa có index đã kiểm chứng. Chạy python -m scripts.build_index trước."
        )
    manifest = json.loads(target.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("Index schema không được hỗ trợ; hãy xây lại index.")
    if require_vectors and not manifest.get("vector_ready"):
        raise ValueError(
            "Snapshot chỉ có BM25; cần xây đủ hai index trước khi kích hoạt."
        )
    for field, checksum_field in (
        ("chunks_path", "chunks_sha256"),
        ("bm25_path", "bm25_sha256"),
    ):
        if manifest.get(checksum_field):
            actual = hashlib.sha256(
                project_path(manifest[field]).read_bytes()
            ).hexdigest()
            if actual != manifest[checksum_field]:
                raise ValueError(f"Index checksum mismatch: {field}")
    return manifest


def matches_filters(meta, domain=None, document_version=None, as_of_date=None):
    if domain and meta.get("domain") != domain:
        return False
    if document_version and meta.get("document_version") != document_version:
        return False
    if as_of_date:
        day = date.fromisoformat(str(as_of_date))
        if not meta.get("validity_verified") or not meta.get("effective_from"):
            return False
        if date.fromisoformat(meta["effective_from"]) > day:
            return False
        if meta.get("effective_to") and day >= date.fromisoformat(meta["effective_to"]):
            return False
    return True
