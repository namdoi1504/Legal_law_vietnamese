# src/pipeline/document_qa_pipeline.py
from dataclasses import dataclass, field
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.bge_reranker import BGEReranker
from src.llm.factory import get_llm_client
from src.parser.document_reader import DocumentReader

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

        # Giới hạn để không overflow context window
        if len(combined_doc) > 6000:
            combined_doc = combined_doc[:6000] + "\n...[nội dung còn lại đã rút gọn]"

        # 2. Tạo retrieval query
        # Ưu tiên câu hỏi của user, nếu không có thì dùng nội dung file
        retrieval_query = user_question if user_question else combined_doc[:300]

        # 3. Retrieve + Rerank điều luật liên quan
        candidates = self.retriever.retrieve(retrieval_query, top_k=15)
        final_chunks = self.reranker.rerank(retrieval_query, candidates, top_k=8)

        if not final_chunks:
            return DocumentQAResult(
                answer="Không tìm thấy quy định pháp luật liên quan trong dữ liệu hiện có để đánh giá văn bản này.",
                citations=[],
                retrieved_chunks=[]
            )

        # 4. Build prompt
        law_context = self._build_law_context(final_chunks)

        user_prompt = f"""VĂN BẢN CẦN KIỂM TRA:
{combined_doc}

QUY ĐỊNH PHÁP LUẬT LIÊN QUAN:
{law_context}

YÊU CẦU: {user_question if user_question else "Kiểm tra toàn bộ văn bản, phát hiện điều khoản vi phạm pháp luật lao động Việt Nam."}
"""

        # 5. Generate
        answer = self.llm.chat(
            system_prompt=DOCUMENT_QA_SYSTEM_PROMPT,
            user_message=user_prompt,
            temperature=0.1  # Thấp để tránh hallucination
        )

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