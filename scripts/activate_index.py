"""Validate and activate an existing snapshot, including rollback to a previous one."""
import argparse
import json
import os
from uuid import uuid4

from configs.settings import project_path, settings
from src.corpus import load_manifest
from src.retriever.bm25_retriever import BM25LegalRetriever
from src.vectordb.qdrant_store import LegalVectorStore


def activate(path):
    manifest = load_manifest(path)
    bm25 = BM25LegalRetriever()
    bm25.load_index(project_path(manifest['bm25_path']))
    if bm25.corpus_version != manifest['corpus_version']:
        raise ValueError('BM25/manifest version mismatch')
    store = LegalVectorStore(collection=manifest['collection'], path=manifest['qdrant_path'], mode=manifest['qdrant_mode'])
    try:
        store.validate_snapshot(bm25.chunks, manifest['corpus_version'])
    finally:
        store.close()
    target = project_path(settings.ACTIVE_INDEX_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f'.{uuid4().hex}.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, target)
    print('Activated corpus:', manifest['corpus_version'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest')
    activate(parser.parse_args().manifest)
