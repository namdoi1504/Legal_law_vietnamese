# src/pipeline/document_qa_pipeline.py
from dataclasses import dataclass, field
from src.retriever.hybrid_retriever import HybridRetriever
from src.reranker.bge_reranker import BGEReranker
from src.llm.factory import get_llm_client
from src.parser.document_reader import DocumentReader

DOCUMENT_QA_SYSTEM_PROMPT = """Ban la chuyen gia phap luat lao dong Viet Nam.

NHIEM VU:
- Doc ky van ban nguoi dung cung cap
- Doi chieu tung dieu khoan voi quy dinh phap luat trong CONTEXT
- Phat hien dieu khoan SAI hoac KHONG PHU HOP voi phap luat Viet Nam
- Giai thich ro tai sao sai va quy dinh dung la gi

QUY TAC BAT BUOC:
1. Chi dua vao CONTEXT phap luat duoc cung cap de danh gia
2. Neu khong tim thay can cu phap ly trong context -> noi ro "Khong tim thay quy dinh lien quan trong du lieu hien co"
3. KHONG duoc bia dat hoac suy dien ngoai context
4. Trich dan dieu luat cu the khi chi ra loi hoac xac nhan dung
5. Tra loi bang tieng Viet, ro rang, co cau truc

DINH DANG TRA LOI BAT BUOC:
## Ket qua kiem tra

OK **Dieu khoan hop le:**
- [liet ke dieu khoan dung + can cu phap ly]

FAIL **Dieu khoan vi pham:**
- [dieu khoan sai] -> [giai thich] -> Can cu: [Dieu X - Ten luat]

? **Khong du can cu de danh gia:**
- [liet ke neu co]
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

        # 1. Extract text tu tat ca file
        all_texts = []
        for file_bytes, filename in file_bytes_list:
            try:
                text = self.reader.read_bytes(file_bytes, filename)
                all_texts.append(f"=== {filename} ===\n{text}")
            except Exception as e:
                return DocumentQAResult(
                    answer="",
                    error=f"Khong the doc file {filename}: {str(e)}"
                )

        combined_doc = "\n\n".join(all_texts)

        # Gioi han de khong overflow context window
        if len(combined_doc) > 6000:
            combined_doc = combined_doc[:6000] + "\n...[noi dung con lai da rut gon]"

        # 2. Tao retrieval query
        # Uu tien cau hoi cua user, neu khong co thi tao query co cau truc
        if user_question:
            retrieval_query = user_question
        else:
            # Tao query phap ly co cau truc thay vi dung raw text
            doc_preview = combined_doc.strip()
            # Bo cac dong qua ngan (thuong la header/so trang)
            meaningful_lines = [
                line for line in doc_preview.split("\n")
                if len(line.strip()) > 20
            ]
            core_text = " ".join(meaningful_lines[:10])[:400]
            retrieval_query = f"kiem tra vi pham phap luat lao dong hop dong lao dong {core_text}"

        # 3. Retrieve + Rerank dieu luat lien quan
        candidates = self.retriever.retrieve(retrieval_query, top_k=20)
        final_chunks = self.reranker.rerank(retrieval_query, candidates, top_k=8)

        if not final_chunks:
            return DocumentQAResult(
                answer="Khong tim thay quy dinh phap luat lien quan trong du lieu hien co de danh gia van ban nay.",
                citations=[],
                retrieved_chunks=[]
            )

        # 4. Build prompt
        law_context = self._build_law_context(final_chunks)

        user_prompt = f"""VAN BAN CAN KIEM TRA:
{combined_doc}

QUY DINH PHAP LUAT LIEN QUAN:
{law_context}

YEU CAU: {user_question if user_question else "Kiem tra toan bo van ban, phat hien dieu khoan vi pham phap luat lao dong Viet Nam."}
"""

        # 5. Generate
        answer = self.llm.chat(
            system_prompt=DOCUMENT_QA_SYSTEM_PROMPT,
            user_message=user_prompt,
            temperature=0.1
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
