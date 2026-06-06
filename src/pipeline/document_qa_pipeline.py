# src/pipeline/document_qa_pipeline.py
import logging
from dataclasses import dataclass, field
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.bge_reranker import BGEReranker
from src.llm.factory import get_llm_client
from src.parser.document_reader import DocumentReader
from src.pipeline.query_rewriter import QueryRewriter
from configs.settings import settings

logger = logging.getLogger(__name__)

# Giới hạn context window gửi lên LLM (ký tự)
# Groq/Gemini thường hỗ trợ 8k–32k token ≈ 16k–64k ký tự tiếng Việt
# Đặt 20000 để an toàn với mọi provider, đồng thời giữ đủ nội dung hợp đồng
DOC_CONTEXT_LIMIT = 20_000

# Kích thước mỗi chunk khi chia tài liệu dài (ký tự)
# LLM sẽ xử lý từng chunk riêng rồi tổng hợp
DOC_CHUNK_SIZE = 18_000

DOCUMENT_QA_SYSTEM_PROMPT = """Bạn là chuyên gia pháp luật lao động Việt Nam.

NHIỆM VỤ:
- Đọc kỹ văn bản người dùng cung cấp
- Đối chiếu từng điều khoản với quy định pháp luật trong CONTEXT
- Phát hiện điều khoản SAI hoặc KHÔNG PHÙ HỢP với pháp luật Việt Nam
- Giải thích rõ tại sao sai và quy định đúng là gì

QUY TẮC BẮT BUỘC:
1. Chỉ dựa vào CONTEXT pháp luật được cung cấp để đánh giá
2. Nếu không tìm thấy căn cứ pháp lý trong context → nói rõ "Không tìm thấy quy định liên quan trong dữ liệu hiện có"
3. KHÔNG được bịa đặt hoặc suy diễn ngoài context
4. Trích dẫn điều luật cụ thể khi chỉ ra lỗi hoặc xác nhận đúng
5. Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc

ĐỊNH DẠNG TRẢ LỜI BẮT BUỘC:
## Kết quả kiểm tra

✅ **Điều khoản hợp lệ:**
- [liệt kê điều khoản đúng + căn cứ pháp lý]

❌ **Điều khoản vi phạm:**
- [điều khoản sai] → [giải thích] → Căn cứ: [Điều X - Tên luật]

⚠️ **Không đủ căn cứ để đánh giá:**
- [liệt kê nếu có]
"""


@dataclass
class DocumentQAResult:
    answer: str
    citations: list[str] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    error: str = None


class DocumentQAPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker = BGEReranker()
        self.llm = get_llm_client()
        self.reader = DocumentReader()
        self.rewriter = QueryRewriter(self.llm)

    def run(
        self,
        file_bytes_list: list[tuple[bytes, str]],
        user_question: str = None
    ) -> DocumentQAResult:

        # 1. Extract text từ tất cả file
        all_texts = []
        for file_bytes, filename in file_bytes_list:
            try:
                text = self.reader.read_bytes(file_bytes, filename)
                all_texts.append(f"=== {filename} ===\n{text}")
            except Exception as e:
                return DocumentQAResult(
                    answer="",
                    error=f"Không thể đọc file {filename}: {str(e)}"
                )

        combined_doc = "\n\n".join(all_texts)
        doc_len = len(combined_doc)
        logger.info(f"[DocumentQA] Tài liệu: {doc_len} ký tự")

        # 2. Tạo retrieval query
        if user_question:
            # Rewrite câu hỏi thông tục → pháp lý (giống rag_pipeline)
            retrieval_query = user_question
            if settings.ENABLE_QUERY_REWRITE:
                retrieval_query, was_rewritten = self.rewriter.rewrite(user_question)
                if was_rewritten:
                    logger.info(f"[DocumentQA] Query rewritten: {retrieval_query!r}")
        else:
            # Không có câu hỏi → tạo query từ nội dung văn bản
            meaningful_lines = [
                line for line in combined_doc.strip().split("\n")
                if len(line.strip()) > 20
            ]
            core_text = " ".join(meaningful_lines[:10])[:400]
            retrieval_query = f"kiểm tra vi phạm pháp luật lao động hợp đồng lao động {core_text}"

        # 3. Retrieve + Rerank điều luật liên quan
        candidates = self.retriever.retrieve(retrieval_query, top_k=20)
        final_chunks = self.reranker.rerank(retrieval_query, candidates, top_k=8)

        if not final_chunks:
            return DocumentQAResult(
                answer="Không tìm thấy quy định pháp luật liên quan trong dữ liệu hiện có để đánh giá văn bản này.",
                citations=[],
                retrieved_chunks=[]
            )

        # 4. Build law context
        law_context = self._build_law_context(final_chunks)

        # 5. Xử lý tài liệu dài: chia chunk, gọi LLM từng chunk, tổng hợp
        if doc_len > DOC_CONTEXT_LIMIT:
            answer = self._run_chunked(combined_doc, law_context, user_question)
        else:
            answer = self._run_single(combined_doc, law_context, user_question)

        citations = [
            f"{c['metadata'].get('article', '')} - "
            f"{c['metadata'].get('article_title', '')} "
            f"({c['metadata'].get('law_name', '')} {c['metadata'].get('law_year', '')})"
            for c in final_chunks
        ]

        return DocumentQAResult(
            answer=answer,
            citations=citations,
            retrieved_chunks=final_chunks
        )

    def _run_single(self, doc_text: str, law_context: str, user_question: str) -> str:
        """Xử lý tài liệu vừa trong 1 lần gọi LLM."""
        user_prompt = f"""VĂN BẢN CẦN KIỂM TRA:
{doc_text}

QUY ĐỊNH PHÁP LUẬT LIÊN QUAN:
{law_context}

YÊU CẦU: {user_question if user_question else "Kiểm tra toàn bộ văn bản, phát hiện điều khoản vi phạm pháp luật lao động Việt Nam."}
"""
        return self.llm.chat(
            system_prompt=DOCUMENT_QA_SYSTEM_PROMPT,
            user_message=user_prompt,
            temperature=0.1
        )

    def _run_chunked(self, doc_text: str, law_context: str, user_question: str) -> str:
        """
        Tài liệu quá dài → chia thành nhiều chunk, xử lý từng chunk,
        sau đó tổng hợp tất cả kết quả thành báo cáo cuối cùng.
        """
        chunks = self._split_doc_into_chunks(doc_text, DOC_CHUNK_SIZE)
        total = len(chunks)
        logger.info(f"[DocumentQA] Tài liệu dài, chia thành {total} chunk để xử lý")

        partial_results = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"[DocumentQA] Xử lý chunk {i}/{total} ({len(chunk)} ký tự)")
            prompt = f"""VĂN BẢN CẦN KIỂM TRA (Phần {i}/{total}):
{chunk}

QUY ĐỊNH PHÁP LUẬT LIÊN QUAN:
{law_context}

YÊU CẦU: {user_question if user_question else "Kiểm tra phần này của văn bản, phát hiện điều khoản vi phạm pháp luật lao động Việt Nam."}
Lưu ý: Đây là phần {i}/{total} của văn bản. Chỉ phân tích nội dung trong phần này.
"""
            result = self.llm.chat(
                system_prompt=DOCUMENT_QA_SYSTEM_PROMPT,
                user_message=prompt,
                temperature=0.1
            )
            partial_results.append(f"### Kết quả phần {i}/{total}:\n{result}")

        # Tổng hợp tất cả phần
        if len(partial_results) == 1:
            return partial_results[0].replace(f"### Kết quả phần 1/1:\n", "")

        synthesis_prompt = f"""Dưới đây là kết quả kiểm tra pháp lý từng phần của một văn bản lao động.
Hãy tổng hợp thành 1 báo cáo duy nhất, loại bỏ trùng lặp, giữ nguyên format:

{chr(10).join(partial_results)}

Hãy tổng hợp thành báo cáo cuối cùng theo đúng định dạng yêu cầu.
"""
        return self.llm.chat(
            system_prompt=DOCUMENT_QA_SYSTEM_PROMPT,
            user_message=synthesis_prompt,
            temperature=0.1
        )

    def _split_doc_into_chunks(self, text: str, chunk_size: int) -> list[str]:
        """
        Chia văn bản thành các chunk theo kích thước, ưu tiên cắt tại
        dòng trống (ranh giới điều khoản) thay vì cắt giữa chừng.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:])
                break

            # Tìm dòng trống gần nhất trước vị trí end để cắt gọn
            cut = text.rfind("\n\n", start, end)
            if cut == -1 or cut == start:
                # Fallback: cắt tại newline đơn
                cut = text.rfind("\n", start, end)
            if cut == -1 or cut == start:
                # Fallback cuối: cắt cứng
                cut = end

            chunks.append(text[start:cut])
            start = cut + 1  # bỏ ký tự newline đầu chunk tiếp theo

        return chunks

    def _build_law_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            meta = chunk["metadata"]
            cite = (
                f"{meta.get('article', '')} - "
                f"{meta.get('article_title', '')} "
                f"({meta.get('law_name', '')} {meta.get('law_year', '')})"
            )
            parts.append(f"[{i}] {cite}\n{chunk['content']}")
        return "\n\n---\n\n".join(parts)
