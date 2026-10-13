# src/llm/base_client.py
from abc import ABC, abstractmethod


class LLMRateLimitError(Exception):
    """Provider quota is temporarily exhausted, not a lack of legal evidence."""

    def __init__(self, retry_after=30):
        self.retry_after = max(1, min(float(retry_after), 3600))
        super().__init__('LLM rate limit exceeded')


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self, system_prompt: str, user_message: str, temperature: float = 0.1
    ) -> str:
        pass
