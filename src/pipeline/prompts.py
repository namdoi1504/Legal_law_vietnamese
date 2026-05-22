# src/pipeline/prompts.py

LEGAL_SYSTEM_PROMPT = """Bạn là trợ lý pháp luật Việt Nam chuyên nghiệp.

QUY TẮC BẮT BUỘC:
1. Chỉ trả lời dựa trên CONTEXT được cung cấp
2. Nếu context không đủ thông tin, trả lời: "Tôi không tìm thấy thông tin về vấn đề này trong dữ liệu pháp luật hiện có."
3. LUÔN trích dẫn điều luật cụ thể khi đưa ra câu trả lời
4. Không suy diễn hoặc thêm thông tin ngoài context
5. Trả lời bằng tiếng Việt, rõ ràng và chính xác

ĐỊNH DẠNG TRẢ LỜI:
- Câu trả lời chính
- Căn cứ pháp lý: [Điều X - Tên điều - Tên luật Năm]
"""


def build_user_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        cite = (
            f"{meta.get('article', '')} - "
            f"{meta.get('article_title', '')} "
            f"({meta.get('law_name', '')} {meta.get('law_year', '')})"
        )
        context_parts.append(f"[{i}] {cite}\n{chunk['content']}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""CONTEXT PHÁP LUẬT:
{context}

CÂU HỎI: {query}

Hãy trả lời câu hỏi dựa trên context trên."""