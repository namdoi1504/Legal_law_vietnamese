import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from configs.settings import settings
from src.parser.labor_law_parser import LaborLawParser
from src.parser.validator import validate_chunks
from src.pipeline.context import build_context, serialize_context, token_upper_bound
from src.pipeline.answers import validate_answer, render_answer
from src.pipeline.rag_pipeline import LegalRAGPipeline, RAGResult
from src.pipeline.document_qa_pipeline import DocumentQAPipeline
from src.retriever.bm25_retriever import BM25LegalRetriever
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.bge_reranker import BGEReranker
from src.corpus import matches_filters
from src.evaluation.metrics import retrieval_metrics


def chunk(
    key="one",
    article="Điều 1",
    content="Nội dung căn cứ pháp luật có thể kiểm chứng.",
    **meta
):
    return {
        "chunk_id": key,
        "content": content,
        "embed_text": content,
        "rerank_score": 5.0,
        "metadata": {
            "chunk_id": key,
            "article": article,
            "law_name": "Luật thử nghiệm",
            "law_year": 2019,
            "document_id": "test",
            "document_version": "v1",
            "domain": "labor",
            **meta,
        },
    }


class ParserTests(unittest.TestCase):
    def test_preserves_lead_and_vietnamese_point(self):
        text = "Điều 1. Tiêu đề\n1. Chỉ áp dụng khi có điều kiện:\na) Trường hợp A.\nđ) Trường hợp Đ."
        rows = LaborLawParser().parse_text(text)
        self.assertEqual([c.metadata["point"] for c in rows], ["a", "đ"])
        self.assertTrue(all("Chỉ áp dụng khi có điều kiện" in c.content for c in rows))
        self.assertEqual(rows[1].metadata["original_content"], "Trường hợp Đ.")
        self.assertEqual(rows[0].metadata["parent_id"], rows[1].metadata["parent_id"])

    def test_quoted_amendments_do_not_create_duplicate_clauses(self):
        text = "Điều 1. Sửa đổi\n1. Sửa như sau:\na) Nội dung:\n“Điều 9. Trích dẫn\n1. Điều kiện\na) A\nđ) Đ.”\nb) Bãi bỏ.\n2. Quy định khác."
        rows = LaborLawParser().parse_text(text)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(c.article_number == "1" for c in rows))
        self.assertIn("đ) Đ.", rows[0].content)
        validate_chunks(rows, [1])

    def test_chapter_does_not_leak_into_previous_article(self):
        rows = LaborLawParser().parse_text(
            "Điều 1. Một\nNội dung.\nChương II\nTiêu đề chương\nĐiều 2. Hai\nNội dung hai."
        )
        self.assertNotIn("Chương", rows[0].content)

    def test_html_wrapped_title(self):
        html = '<div class="content1"><p>Điều 1. Tiêu đề\nxuống dòng</p><p>1. Nội dung.</p></div>'
        row = LaborLawParser().parse_html(html)[0]
        self.assertEqual(row.article_title, "Tiêu đề xuống dòng")
        self.assertNotIn("Tiêu đề", row.content)

    def test_invalid_source_fails(self):
        with self.assertRaises(ValueError):
            LaborLawParser().parse_html("<html>access denied</html>")

    def test_id_stability(self):
        p = LaborLawParser()
        text = "Điều 1. Một\nNội dung."
        self.assertEqual(p.parse_text(text)[0].chunk_id, p.parse_text(text)[0].chunk_id)
        self.assertNotEqual(
            p.parse_text(text)[0].chunk_id, p.parse_text(text + " Mới.")[0].chunk_id
        )


class RetrievalTests(unittest.TestCase):
    def test_bm25_filters_before_top_k(self):
        rows = [
            chunk("a", content="lương lương lương", domain="other"),
            chunk("b", content="lương thấp", domain="labor"),
            chunk("c", content="nghỉ thai sản", domain="labor"),
        ]
        r = BM25LegalRetriever()
        r.build_index(rows, "v1")
        result = r.search("lương", 1, domain="labor")
        self.assertEqual(result[0]["metadata"]["chunk_id"], "b")
        self.assertEqual(r.search("", 1), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bm25.pkl"
            r.save_index(path)
            loaded = BM25LegalRetriever()
            loaded.load_index(path)
            self.assertEqual(loaded.corpus_version, "v1")

    def test_time_filter_excludes_unverified_and_expired(self):
        meta = {
            "effective_from": "2021-01-01",
            "effective_to": "2023-01-01",
            "validity_verified": True,
        }
        self.assertTrue(matches_filters(meta, as_of_date="2022-01-01"))
        self.assertFalse(matches_filters(meta, as_of_date="2023-01-01"))
        self.assertFalse(matches_filters({}, as_of_date="2022-01-01"))

    def test_fusion_does_not_double_count_duplicate(self):
        a, b = chunk("a"), chunk("b")
        out = HybridRetriever._fuse([[a, a], [b]], 5)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["fusion_score"], out[1]["fusion_score"])

    def test_hybrid_preserves_original_and_filters_both_branches(self):
        e = Mock()
        e.embed_single.return_value = [1, 0]
        store = Mock()
        store.search.return_value = [chunk()]
        bm = Mock()
        bm.search.return_value = [chunk()]
        r = HybridRetriever(e, store, bm)
        r.retrieve(
            "gốc", domain="labor", rewritten_query="mới", as_of_date="2022-01-01"
        )
        self.assertEqual([c.args[0] for c in bm.search.call_args_list], ["gốc", "mới"])
        self.assertTrue(
            all(c.kwargs["domain"] == "labor" for c in bm.search.call_args_list)
        )
        self.assertTrue(
            all(
                c.kwargs["domain_filter"] == "labor"
                for c in store.search.call_args_list
            )
        )

    def test_reranker_keeps_high_scoring_siblings(self):
        model = Mock()
        model.predict.return_value = [9, 8, 1]
        rows = [chunk("a"), chunk("b"), chunk("c", "Điều 2")]
        with patch.object(settings, "RERANK_MAX_PER_ARTICLE", 0):
            out = BGEReranker(model).rerank("question", rows, 2)
        self.assertEqual([c["metadata"]["chunk_id"] for c in out], ["a", "b"])
        self.assertEqual(rows[0]["rerank_score"], 5)

    def test_empty_reranker_does_not_call_model(self):
        model = Mock()
        self.assertEqual(BGEReranker(model).rerank("q", []), [])
        model.predict.assert_not_called()

    def test_recall_requires_all_evidence_groups(self):
        item = {
            "expected_evidence_groups": [
                [{"article": "Điều 1", "clause": "1"}],
                [{"article": "Điều 1", "clause": "2"}],
            ]
        }
        self.assertEqual(retrieval_metrics(item, [chunk(clause="1")])["recall"], 0.5)
        self.assertEqual(retrieval_metrics(item, [chunk(clause="3")])["hit"], 0)


class ContextTests(unittest.TestCase):
    def test_parent_expansion_deduplicates_siblings(self):
        rows = [
            chunk(
                "a",
                point="a",
                clause="1",
                parent_id="parent",
                parent_content="Điều kiện chung. a) A. đ) Đ.",
            ),
            chunk(
                "b",
                point="đ",
                clause="1",
                parent_id="parent",
                parent_content="Điều kiện chung. a) A. đ) Đ.",
            ),
        ]
        out = build_context(rows, 2000)
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0]["metadata"]["point"])
        self.assertEqual(out[0]["metadata"]["chunk_id"], "parent")

    def test_context_respects_budget_and_does_not_cut_evidence(self):
        rows = [
            chunk("a", content="x" * 1000),
            chunk("b", content="Quy định ngắn hơn."),
        ]
        out = build_context(rows, 300)
        self.assertLessEqual(token_upper_bound(serialize_context(out)), 300)
        self.assertEqual(out[0]["content"], "Quy định ngắn hơn.")


class AnswerTests(unittest.TestCase):
    def setUp(self):
        self.context = build_context([chunk()], 2000)
        self.payload = {
            "status": "answered",
            "claims": [
                {
                    "text": "Nhận định.",
                    "evidence": [
                        {"source_id": "S1", "quote": "Nội dung căn cứ pháp luật"}
                    ],
                }
            ],
            "missing_information": [],
        }

    def test_only_used_sources_are_citations(self):
        context = self.context + [{**chunk("two"), "source_id": "S2"}]
        parsed = validate_answer(json.dumps(self.payload), context)
        text, citations, sources = render_answer(parsed, context)
        self.assertEqual(len(citations), 1)
        self.assertIn("[S1]", text)
        self.assertEqual(sources[0]["source_id"], "S1")

    def test_rejects_fabricated_quote_and_id(self):
        for evidence in (
            {"source_id": "S99", "quote": "Nội dung căn cứ pháp luật"},
            {"source_id": "S1", "quote": "Trích dẫn hoàn toàn bịa đặt"},
        ):
            self.payload["claims"][0]["evidence"] = [evidence]
            with self.assertRaises(ValueError):
                validate_answer(json.dumps(self.payload), self.context)

    def test_needs_information_requires_question(self):
        with self.assertRaises(ValueError):
            validate_answer('{"status":"needs_information"}', self.context)

    def test_pipeline_rejects_invalid_generation(self):
        llm = Mock()
        llm.chat.return_value = '{"status":"answered","claims":[]}'
        r = Mock()
        r.retrieve.return_value = [chunk()]
        rank = Mock()
        rank.rerank.return_value = [chunk()]
        p = LegalRAGPipeline(r, rank, llm)
        with patch.object(settings, "ENABLE_QUERY_REWRITE", False):
            result = p.run("Nội dung câu hỏi?")
        self.assertEqual(result.status, "validation_failed")
        self.assertEqual(result.citations, [])

    def test_empty_context_never_calls_llm(self):
        llm = Mock()
        r = Mock()
        r.retrieve.return_value = []
        rank = Mock()
        rank.rerank.return_value = []
        with patch.object(settings, "ENABLE_QUERY_REWRITE", False):
            result = LegalRAGPipeline(r, rank, llm).run("Câu hỏi?")
        self.assertEqual(result.status, "insufficient_evidence")
        llm.chat.assert_not_called()


class DocumentTests(unittest.TestCase):
    def test_split_preserves_every_character(self):
        for text in ("x" * 100, "Điều 1. A\nĐiều 2. B\n\nĐoạn cuối.", "abc\ndefghijkl"):
            chunks = DocumentQAPipeline._split_doc_into_chunks(text, 7)
            self.assertEqual("".join(chunks), text)
            self.assertTrue(all(len(c) <= 7 for c in chunks))

    def test_each_section_has_its_own_retrieval(self):
        reader = Mock()
        reader.read_bytes.return_value = "Điều 1. Tiền lương.\nĐiều 2. Nghỉ phép."
        p = Mock()
        p.retriever.retrieve.return_value = [chunk()]
        p.reranker.rerank.return_value = [chunk()]
        p.answer_from_chunks.return_value = RAGResult(answer="Chưa đủ căn cứ.")
        with patch.object(settings, "DOCUMENT_SECTION_CHARS", 20):
            out = DocumentQAPipeline(p, reader).run([(b"file", "test.docx")])
        self.assertGreater(len(out.sections), 1)
        self.assertEqual(p.retriever.retrieve.call_count, len(out.sections))
        self.assertIn("Nghỉ phép", p.retriever.retrieve.call_args.args[0])

    def test_short_contract_is_split_at_each_clause(self):
        text = "Điều 1. Lương\nĐiều 2. Nghỉ phép\nĐiều 3. Chấm dứt"
        self.assertEqual(len(DocumentQAPipeline._split_doc_into_chunks(text, 2500)), 3)
        self.assertEqual(len(DocumentQAPipeline._split_doc_into_chunks(text.upper(), 2500)), 3)

    def test_provider_failure_stops_document_and_reports_remaining_sections(self):
        reader = Mock()
        reader.read_bytes.return_value = "Điều 1. A\nĐiều 2. B\nĐiều 3. C"
        pipeline = Mock()
        pipeline.retriever.retrieve.return_value = []
        pipeline.reranker.rerank.return_value = []
        pipeline.answer_from_chunks.return_value = RAGResult(answer="Hết quota.", status="rate_limited")
        result = DocumentQAPipeline(pipeline, reader).run([(b"file", "sample.docx")])
        self.assertEqual(result.status, "partial")
        self.assertEqual(result.unprocessed_sections, 2)
        self.assertEqual(pipeline.answer_from_chunks.call_count, 1)

    def test_document_limits_are_reported(self):
        reader = Mock()
        reader.read_bytes.return_value = "a" * 30
        p = Mock()
        p.retriever.retrieve.return_value = []
        p.reranker.rerank.return_value = []
        p.answer_from_chunks.return_value = RAGResult(answer="Chưa đủ căn cứ.")
        with patch.object(settings, "DOCUMENT_SECTION_CHARS", 10), patch.object(
            settings, "MAX_DOCUMENT_SECTIONS", 1
        ):
            out = DocumentQAPipeline(p, reader).run([(b"file", "test.docx")])
        self.assertEqual(out.unprocessed_sections, 2)
        self.assertEqual(out.status, "partial")


class QdrantTests(unittest.TestCase):
    def test_stable_ids_idempotence_and_no_document_overwrite(self):
        from qdrant_client import QdrantClient
        from src.vectordb.qdrant_store import LegalVectorStore

        store = LegalVectorStore(client=QdrantClient(":memory:"), collection="test")
        try:
            with patch.object(settings, "VECTOR_SIZE", 3):
                store.create_collection()
            store.insert_chunks([chunk("doc1")], [[1.0, 0.0, 0.0]])
            store.insert_chunks([chunk("doc1")], [[1.0, 0.0, 0.0]])
            store.insert_chunks([chunk("doc2")], [[0.0, 1.0, 0.0]])
            self.assertEqual(store.client.count("test").count, 2)
            with self.assertRaises(ValueError):
                store.insert_chunks([chunk()], [])
            with self.assertRaises(ValueError):
                store.insert_chunks([chunk(), chunk()], [[1.0, 0.0, 0.0]] * 2)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
