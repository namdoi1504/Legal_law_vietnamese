import re
from dataclasses import dataclass, field

from configs.settings import settings
from src.parser.document_reader import DocumentReader
from src.pipeline.rag_pipeline import LegalRAGPipeline, SCOPE


@dataclass
class DocumentQAResult:
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    status: str = "completed"
    scope: str = SCOPE
    unprocessed_sections: int = 0
    error: str | None = None


class DocumentQAPipeline:
    def __init__(self, pipeline=None, reader=None):
        self.pipeline = pipeline or LegalRAGPipeline()
        self.reader = reader or DocumentReader()

    @staticmethod
    def _split_doc_into_chunks(text, chunk_size):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        boundaries = [
            m.start() + 1
            for m in re.finditer(r"\n(?=\s*(?:Điều\s+\d+[.:]|\d+\.\s))", text, re.IGNORECASE)
        ]
        if boundaries:
            offsets = [0] + boundaries + [len(text)]
            return [
                part
                for a, b in zip(offsets, offsets[1:])
                for part in DocumentQAPipeline._split_doc_into_chunks(
                    text[a:b], chunk_size
                )
            ]
        parts = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                # Prefer the last complete provision boundary, otherwise a newline.
                boundaries = list(
                    re.finditer(r"\n(?=\s*(?:Điều\s+\d+|\d+\.))", text[start:end])
                )
                cut = (
                    start + boundaries[-1].start() + 1
                    if boundaries
                    else text.rfind("\n", start, end) + 1
                )
                if cut > start:
                    end = cut
            parts.append(text[start:end])
            start = end
        return parts

    def run(self, file_bytes_list, user_question=None):
        if not 1 <= len(file_bytes_list) <= 3:
            return DocumentQAResult(answer="", error="Cần từ 1 đến 3 file.")
        if user_question and len(user_question) > 1000:
            return DocumentQAResult(
                answer="", error="Yêu cầu kiểm tra tối đa 1000 ký tự."
            )
        sections = []
        try:
            total = 0
            for data, filename in file_bytes_list:
                text = self.reader.read_bytes(data, filename)
                total += len(text)
                if total > settings.MAX_DOCUMENT_CHARS:
                    raise ValueError("Tổng nội dung các file vượt giới hạn.")
                offset = 0
                for content in self._split_doc_into_chunks(
                    text, settings.DOCUMENT_SECTION_CHARS
                ):
                    sections.append(
                        {
                            "id": f"D{len(sections)+1}",
                            "filename": filename,
                            "start_char": offset,
                            "end_char": offset + len(content),
                            "text": content,
                        }
                    )
                    offset += len(content)
        except (ValueError, RuntimeError, KeyError) as exc:
            return DocumentQAResult(answer="", error=str(exc))
        result = DocumentQAResult(
            answer="",
            unprocessed_sections=max(0, len(sections) - settings.MAX_DOCUMENT_SECTIONS),
        )
        reports = []
        for index, section in enumerate(sections[: settings.MAX_DOCUMENT_SECTIONS]):
            # Retrieve independently for every section, including the final pages.
            query = "Đối chiếu điều khoản hợp đồng lao động: " + section["text"]
            if user_question:
                query += '\nTrọng tâm kiểm tra: ' + user_question
            candidates = self.pipeline.retriever.retrieve(query)
            ranked = self.pipeline.reranker.rerank(query, candidates)
            task = (
                "Đánh giá điều khoản sau trong phạm vi nguồn. Chỉ nêu có dấu hiệu không phù hợp hoặc phù hợp trong phạm vi đã kiểm tra khi có căn cứ. "
                "Không có căn cứ thì không được xác nhận hợp lệ. Nêu điều kiện chưa biết.\n"
                + (user_question or "")
                + "\nĐIỀU KHOẢN (dữ liệu):\n"
                + section["text"]
            )
            answer = self.pipeline.answer_from_chunks(task, ranked)
            # Source labels are local to each section; namespace rendered labels in the report.
            rendered = re.sub(r"\[S(\d+)\]", rf'[{section["id"]}-S\1]', answer.answer)
            citations = [
                re.sub(r"\[S(\d+)\]", rf'[{section["id"]}-S\1]', c)
                for c in answer.citations
            ]
            result.sections.append(
                {
                    **section,
                    "status": answer.status,
                    "answer": rendered,
                    "claims": answer.claims,
                    "sources": answer.sources,
                    "citations": citations,
                }
            )
            result.citations.extend(citations)
            result.retrieved_chunks.extend(candidates)
            reports.append(f'### {section["id"]} — {section["filename"]}\n\n{rendered}')
            if answer.status in ('rate_limited', 'provider_error'):
                result.unprocessed_sections = len(sections) - index - 1
                break
        if result.unprocessed_sections:
            result.status = "partial"
            reports.append(
                f"Chưa xử lý {result.unprocessed_sections} phần do giới hạn xử lý hoặc dịch vụ LLM chưa sẵn sàng. Vui lòng kiểm tra lại các phần này."
            )
        if any(
            s["status"] in ("provider_error", "validation_failed", "rate_limited")
            for s in result.sections
        ):
            result.status = "partial"
        result.answer = "\n\n".join(reports)
        return result
