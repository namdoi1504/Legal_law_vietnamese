import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from configs.settings import settings
from src.corpus import load_manifest
from src.evaluation.metrics import retrieval_metrics
from src.pipeline.context import build_context
from src.pipeline.query_rewriter import QueryRewriter, expand_legal_query, reranking_query
from src.pipeline.rag_pipeline import LegalRAGPipeline
from test_rag import chunk


class SnapshotTests(unittest.TestCase):
    def test_manifest_checksums_and_incomplete_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'chunks.json'
            data.write_text('[]', encoding='utf-8')
            manifest = {'schema_version':2, 'vector_ready':True, 'chunks_path':str(data),
                        'chunks_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            self.assertTrue(load_manifest(str(path))['vector_ready'])
            data.write_text('[1]', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_manifest(str(path))
            manifest.pop('chunks_sha256')
            manifest['vector_ready'] = False
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError): load_manifest(str(path))
            self.assertFalse(load_manifest(str(path), require_vectors=False)['vector_ready'])

    def test_vector_snapshot_detects_changed_content(self):
        from qdrant_client import QdrantClient
        from src.vectordb.qdrant_store import LegalVectorStore
        store = LegalVectorStore(client=QdrantClient(':memory:'), collection='test')
        try:
            with patch.object(settings, 'VECTOR_SIZE', 3): store.create_collection()
            original = chunk('one', corpus_version='v1')
            store.insert_chunks([original], [[1.,0.,0.]])
            store.validate_snapshot([original], 'v1')
            with self.assertRaisesRegex(ValueError, 'version'):
                store.validate_snapshot([original], 'v2')
            with self.assertRaisesRegex(ValueError, 'content'):
                store.validate_snapshot([chunk('one', content='Changed')], 'v1')
        finally:
            store.close()

    def test_context_recall_counts_points_in_expanded_parent(self):
        source = chunk('point', article='Điều 98', clause='1', point='a', parent_id='parent',
                       parent_content='Điều kiện:\na) Ngày thường.\nb) Ngày nghỉ.')
        context = build_context([source], 2000)
        item = {'expected_evidence_groups':[[{'article':'Điều 98','clause':'1','point':'b'}]]}
        self.assertEqual(retrieval_metrics(item, context)['recall'], 1)
        item['expected_evidence_groups'][0][0]['point'] = 'đ'
        self.assertEqual(retrieval_metrics(item, context)['recall'], 0)

    def test_low_scoring_document_context_never_calls_llm(self):
        llm = Mock()
        pipeline = LegalRAGPipeline(Mock(), Mock(), llm)
        source = chunk(); source['rerank_score'] = -100
        result = pipeline.answer_from_chunks('Đánh giá điều khoản', [source])
        self.assertEqual(result.status, 'insufficient_evidence')
        llm.chat.assert_not_called()


class QueryTests(unittest.TestCase):
    def test_aliases_preserve_facts_and_negation(self):
        query = 'không được trừ lương 500000 đồng khi đi muộn?'
        expanded = expand_legal_query(query)
        self.assertTrue(expanded.startswith(query))
        self.assertIn('xử lý kỷ luật', expanded)
        self.assertNotIn('Điều 127', expanded)

    def test_aliases_work_without_accents(self):
        self.assertIn('thai sản', expand_legal_query('nghi de duoc may thang'))

    def test_alias_rewrite_needs_no_api(self):
        llm = Mock()
        query, rewritten = QueryRewriter(llm).rewrite('bị đuổi việc oan')
        self.assertTrue(rewritten)
        self.assertIn('sa thải', query)
        self.assertIn('chấm dứt hợp đồng', query)
        llm.chat.assert_not_called()

    def test_llm_rewrite_cannot_add_numbers_or_remove_negation(self):
        llm = Mock()
        query = 'công ty không nhận tôi thì sao'
        for answer in ('công ty nhận tôi thì sao', 'công ty không nhận tôi theo luật 2019'):
            llm.chat.return_value = answer
            self.assertEqual(QueryRewriter(llm).rewrite(query), (query,False))

    def test_reranking_preserves_original_question(self):
        self.assertEqual(reranking_query('gốc','mới'), 'gốc\nmới')
        self.assertEqual(reranking_query('gốc','gốc mở rộng'), 'gốc mở rộng')


if __name__ == '__main__': unittest.main()
