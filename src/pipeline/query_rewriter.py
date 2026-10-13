# src/pipeline/query_rewriter.py
import logging
import re
import unicodedata
from configs.settings import settings
from src.pipeline.prompts import QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)

# Từ khóa pháp lý — nếu query đã có các từ này thì KHÔNG cần rewrite
LEGAL_KEYWORDS = {
    "điều",
    "khoản",
    "bộ luật",
    "luật lao động",
    "nghị định",
    "thông tư",
    "hợp đồng lao động",
    "đơn phương chấm dứt",
    "thử việc",
    "thai sản",
    "tiền lương",
    "làm thêm giờ",
    "kỷ luật lao động",
    "sa thải",
    "thôi việc",
    "người sử dụng lao động",
    "người lao động",
    "nội quy lao động",
    "thỏa ước",
    "tranh chấp lao động",
    "đình công",
    "bảo hiểm xã hội",
    "trợ cấp thôi việc",
    "trợ cấp mất việc",
    "tạm hoãn",
    "tạm đình chỉ",
    "quyền đơn phương",
    "vi phạm pháp luật",
    "vô hiệu",
    "hiệu lực",
}

# Độ dài tối thiểu để xét rewrite — câu quá ngắn (< 5 từ) không đủ ngữ cảnh
MIN_WORDS_FOR_REWRITE = 3


def expand_legal_query(query):
    """Append search synonyms; never replace facts or assert a legal conclusion."""
    if not settings.ENABLE_QUERY_ALIASES:
        return query
    folded = unicodedata.normalize('NFD', query.lower().replace('đ', 'd'))
    folded = ''.join(c for c in folded if unicodedata.category(c) != 'Mn')
    aliases = (
        ('duoi viec', 'sa thải chấm dứt hợp đồng lao động'),
        ('viec oan', 'trái pháp luật bồi thường'),
        ('tru luong', 'khấu trừ tiền lương cắt lương phạt tiền xử lý kỷ luật'),
        ('nghi de', 'nghỉ thai sản'),
    )
    additions = [value for phrase,value in aliases if re.search(r'(?<!\w)'+re.escape(phrase)+r'(?!\w)',folded)]
    return query + ('\nTừ khóa tra cứu: ' + '; '.join(additions) if additions else '')


def reranking_query(original, rewritten):
    return rewritten if rewritten.startswith(original) else original + '\n' + rewritten


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

        expanded = expand_legal_query(query)
        if expanded != query:
            return expanded, True

        # Skip nếu query đã dùng thuật ngữ pháp lý
        if self._is_formal_query(query):
            logger.debug('Query rewrite skipped: formal')
            return query, False

        # Skip nếu query quá ngắn
        if len(query.split()) < MIN_WORDS_FOR_REWRITE:
            logger.debug('Query rewrite skipped: short')
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
                    "Rewrite rejected by length validation; using original query"
                )
                return query, False

            if re.findall(r"\d+(?:[.,]\d+)?", query) != re.findall(
                r"\d+(?:[.,]\d+)?", rewritten
            ):
                return query, False
            if any(
                word in query.lower() and word not in rewritten.lower()
                for word in ("không", "chưa")
            ):
                return query, False

            logger.debug("Query rewritten")
            return rewritten, True

        except Exception as e:
            # Fallback an toàn — pipeline không bị crash
            logger.warning(
                "Rewrite unavailable (%s); using original query", type(e).__name__
            )
            return query, False

    def _is_formal_query(self, query: str) -> bool:
        """Trả về True nếu query đã chứa từ khóa pháp lý chuyên ngành."""
        query_lower = query.lower()
        return any(kw in query_lower for kw in LEGAL_KEYWORDS)
