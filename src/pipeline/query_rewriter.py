# src/pipeline/query_rewriter.py
import logging
from src.pipeline.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)

# Từ khóa pháp lý — nếu query đã có các từ này thì KHÔNG cần rewrite
LEGAL_KEYWORDS = {
    "điều", "khoản", "bộ luật", "luật lao động", "nghị định", "thông tư",
    "hợp đồng lao động", "đơn phương chấm dứt", "thử việc", "thai sản",
    "tiền lương", "làm thêm giờ", "kỷ luật lao động", "sa thải", "thôi việc",
    "người sử dụng lao động", "người lao động", "nội quy lao động",
    "thỏa ước", "tranh chấp lao động", "đình công", "bảo hiểm xã hội",
    "trợ cấp thôi việc", "trợ cấp mất việc", "tạm hoãn", "tạm đình chỉ",
    "quyền đơn phương", "vi phạm pháp luật", "vô hiệu", "hiệu lực",
}

# Độ dài tối thiểu để xét rewrite — câu quá ngắn (< 5 từ) không đủ ngữ cảnh
MIN_WORDS_FOR_REWRITE = 3


class QueryRewriter:
    """
    Dịch câu hỏi ngôn ngữ thông tục sang ngôn ngữ pháp lý
    trước khi đưa vào retriever, giúp tăng recall với các
    truy vấn không dùng thuật ngữ chuyên ngành.
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: instance của BaseLLMClient (Groq / Gemini / Ollama)
        """
        self.llm = llm_client

    def rewrite(self, query: str) -> tuple[str, bool]:
        """
        Rewrite query nếu cần thiết.

        Returns:
            (rewritten_query, was_rewritten)
            - rewritten_query: câu hỏi đã được viết lại (hoặc query gốc nếu không cần)
            - was_rewritten: True nếu đã rewrite, False nếu dùng query gốc
        """
        query = query.strip()

        if not query:
            return query, False

        # Skip nếu query đã dùng thuật ngữ pháp lý
        if self._is_formal_query(query):
            logger.debug(f"[QueryRewriter] Skip (formal): {query!r}")
            return query, False

        # Skip nếu query quá ngắn
        if len(query.split()) < MIN_WORDS_FOR_REWRITE:
            logger.debug(f"[QueryRewriter] Skip (too short): {query!r}")
            return query, False

        try:
            rewritten = self.llm.chat(
                system_prompt="",
                user_message=QUERY_REWRITE_PROMPT.format(query=query),
                temperature=0.0,
            )
            rewritten = rewritten.strip()

            # Sanity check: nếu rewritten rỗng hoặc quá khác (> 5x độ dài), fallback
            if not rewritten or len(rewritten) > len(query) * 5:
                logger.warning(
                    f"[QueryRewriter] Rewrite suspicious, fallback to original. "
                    f"Original: {query!r}, Rewritten: {rewritten!r}"
                )
                return query, False

            logger.info(
                f"[QueryRewriter] {query!r}\n"
                f"             → {rewritten!r}"
            )
            return rewritten, True

        except Exception as e:
            # Fallback an toàn — pipeline không bị crash
            logger.warning(f"[QueryRewriter] LLM call failed, using original query. Error: {e}")
            return query, False

    def _is_formal_query(self, query: str) -> bool:
        """Trả về True nếu query đã chứa từ khóa pháp lý chuyên ngành."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in LEGAL_KEYWORDS)
