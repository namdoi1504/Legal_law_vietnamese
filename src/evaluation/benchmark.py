"""Reproducible retrieval and end-to-end reports. API calls require --mode pipeline."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import platform
import statistics
import subprocess
import time

from configs.settings import ROOT, project_path, settings
from src.corpus import load_manifest
from src.evaluation.metrics import retrieval_metrics


def summarize(cases):
    metrics = {}
    for key in ("hit", "recall", "mrr", "ndcg"):
        values = [
            c["retrieval"][key]
            for c in cases
            if c.get("retrieval", {}).get(key) is not None
        ]
        metrics[key] = sum(values) / len(values) if values else None
        metrics[key + "_samples"] = len(values)
    times = sorted(c["latency_ms"] for c in cases)
    metrics.update(
        count=len(cases),
        errors=sum(bool(c.get("error")) for c in cases),
        p50_ms=statistics.median(times) if times else None,
        p95_ms=times[min(len(times) - 1, int(len(times) * 0.95))] if times else None,
    )
    return metrics


def run(dataset, mode, output, limit=200, manifest_path=None, resume=False, retry_errors=False, min_interval=0):
    path = project_path(dataset)
    questions = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:limit]
    manifest = load_manifest(manifest_path, require_vectors=mode != "bm25")
    if mode == "bm25":
        from src.retriever.bm25_retriever import BM25LegalRetriever

        engine = BM25LegalRetriever()
        engine.load_index(project_path(manifest["bm25_path"]))
    else:
        from src.retriever.hybrid_retriever import HybridRetriever

        if manifest["qdrant_mode"] == "local":
            # Evaluation owns a memory index, leaving the running app's file lock alone.
            import numpy as np
            from qdrant_client import QdrantClient
            from src.vectordb.qdrant_store import LegalVectorStore
            from src.retriever.bm25_retriever import BM25LegalRetriever

            bm25 = BM25LegalRetriever()
            bm25.load_index(project_path(manifest["bm25_path"]))
            key = hashlib.sha256(
                (manifest["corpus_version"] + manifest["embedding_model"]).encode()
            ).hexdigest()
            vectors_path = project_path(
                manifest.get("embeddings_path", f"data/embedding_cache/{key}.npy")
            )
            if (
                manifest.get("embeddings_sha256")
                and hashlib.sha256(vectors_path.read_bytes()).hexdigest()
                != manifest["embeddings_sha256"]
            ):
                raise ValueError("Embedding cache checksum mismatch")
            vectors = np.load(vectors_path, allow_pickle=False)
            store = LegalVectorStore(
                client=QdrantClient(":memory:"), collection="evaluation"
            )
            store.create_collection()
            store.insert_chunks(bm25.chunks, vectors.tolist())
            engine = HybridRetriever(vector_store=store, bm25=bm25, manifest=manifest)
        else:
            engine = HybridRetriever(manifest=manifest)
    reranker = None
    pipeline = None
    if mode in ("hybrid", "pipeline"):
        from src.reranker.bge_reranker import BGEReranker

        reranker = BGEReranker()
    if mode == "pipeline":
        from src.pipeline.rag_pipeline import LegalRAGPipeline

        pipeline = LegalRAGPipeline(retriever=engine, reranker=reranker)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--no-ext-diff"], stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
        diff = b""
    from src.pipeline.prompts import PROMPT_VERSION

    source_digest = hashlib.sha256()
    for directory in ('src', 'configs', 'app', 'scripts'):
        for source in sorted((ROOT / directory).rglob('*.py')):
            source_digest.update(source.relative_to(ROOT).as_posix().encode())
            source_digest.update(source.read_bytes())
    source_hash = source_digest.hexdigest()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "python": platform.python_version(),
        "git_revision": revision,
        "working_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "source_tree_sha256": source_hash,
        "dataset": str(path),
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "corpus_version": manifest["corpus_version"],
        "prompt_version": PROMPT_VERSION,
        "settings": {
            key: getattr(settings, key)
            for key in (
                "EMBED_MODEL",
                "RERANKER_MODEL",
                "LLM_PROVIDER",
                "GROQ_MODEL",
                "TOP_K_RERANK",
                "RETRIEVAL_CANDIDATES",
                "ENABLE_QUERY_REWRITE",
                "ENABLE_QUERY_ALIASES",
                "RERANKER_MIN_SCORE",
                "RERANK_MAX_PER_ARTICLE",
                "CONTEXT_TOKEN_BUDGET",
            )
        },
        "limitations": [
            "Labels are not independently reviewed.",
            "Citation checks verify identity and verbatim quotes, not semantic correctness.",
        ],
        "cases": [],
    }
    output = project_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if resume and output.exists():
        previous = json.loads(output.read_text(encoding='utf-8'))
        for key in ('mode', 'dataset_sha256', 'corpus_version', 'settings'):
            if previous.get(key) != report.get(key):
                raise ValueError(f'Cannot resume a different experiment: {key}')
        report['cases'] = previous['cases']
        report['resumed_from'] = previous.get('created_at')
    completed = {c['id']: c for c in report['cases']}
    last_start = None
    try:
        for item in questions:
            previous_case = completed.get(item['id'])
            if previous_case is not None and not (retry_errors and previous_case.get('error')):
                continue
            if last_start is not None:
                remaining = min_interval - (time.perf_counter() - last_start)
                if remaining > 0:
                    time.sleep(remaining)
            start = time.perf_counter()
            last_start = start
            case = {
                "id": item["id"],
                "question": item["question"],
                "type": item.get("type", "unknown"),
                "source_tree_sha256": source_hash,
            }
            try:
                if mode == "pipeline":
                    result = pipeline.run(
                        item["question"], as_of_date=item.get("as_of_date")
                    )
                    ranked = result.reranked_chunks
                    case["result"] = asdict(result)
                    case["context_retrieval"] = retrieval_metrics(
                        item, result.context_chunks
                    )
                    case["error"] = result.error
                    case["human_review"] = {
                        "correctness": None,
                        "completeness": None,
                        "citation_support": None,
                        "notes": "",
                    }
                    if item.get("answerability") == "unanswerable":
                        case["correct_abstention"] = (
                            result.status == "insufficient_evidence"
                        )
                    elif item.get("answerability") == "needs_information":
                        case["asks_for_information"] = result.status in (
                            "needs_information",
                            "partial",
                        )
                elif mode == "bm25":
                    ranked = engine.search(
                        item["question"], top_k=settings.TOP_K_RERANK
                    )
                elif mode == "vector":
                    ranked = engine.vector_store.search(
                        engine.embedder.embed_single(item["question"]),
                        top_k=settings.TOP_K_RERANK,
                    )
                else:
                    from src.pipeline.query_rewriter import expand_legal_query
                    candidates = engine.retrieve(item["question"])
                    ranked = reranker.rerank(expand_legal_query(item["question"]), candidates)
                case["retrieval"] = retrieval_metrics(item, ranked)
                case["retrieved"] = [
                    {
                        "chunk_id": c["metadata"]["chunk_id"],
                        "article": c["metadata"].get("article"),
                        "clause": c["metadata"].get("clause"),
                        "point": c["metadata"].get("point"),
                        "score": c.get("rerank_score", c.get("score")),
                    }
                    for c in ranked
                ]
            except Exception as exc:
                case["error"] = type(exc).__name__
                case["retrieval"] = {"hit": 0, "recall": 0, "mrr": 0, "ndcg": 0}
            case["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
            if previous_case is not None:
                case['previous_errors'] = previous_case.get('previous_errors', []) + [previous_case['error']]
            completed[item['id']] = case
            report['cases'] = [completed[q['id']] for q in questions if q['id'] in completed]
            report["summary"] = summarize(report["cases"])
            report["by_type"] = {
                kind: summarize([c for c in report["cases"] if c["type"] == kind])
                for kind in {c["type"] for c in report["cases"]}
            }
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(
                f"{len(report['cases'])}/{len(questions)} {item['id']} hit={case['retrieval']['hit']} error={case.get('error')}",
                flush=True,
            )
    finally:
        if hasattr(engine, "vector_store"):
            engine.vector_store.close()
    report['summary'] = summarize(report['cases'])
    print(json.dumps(report["summary"], ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/evaluation/dev.jsonl")
    parser.add_argument(
        "--mode", choices=["bm25", "vector", "hybrid", "pipeline"], default="bm25"
    )
    parser.add_argument("--output", default="reports/evaluation/latest.json")
    parser.add_argument("--manifest")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-errors', action='store_true')
    parser.add_argument('--min-interval', type=float, default=0, help='Minimum seconds between case starts, for API quotas')
    args = parser.parse_args()
    if not 1 <= args.limit <= 200:
        parser.error("--limit must be between 1 and 200")
    if not 0 <= args.min_interval <= 60:
        parser.error('--min-interval must be between 0 and 60 seconds')
    if args.retry_errors and not args.resume:
        parser.error('--retry-errors requires --resume')
    run(args.dataset, args.mode, args.output, args.limit, args.manifest, args.resume, args.retry_errors, args.min_interval)
