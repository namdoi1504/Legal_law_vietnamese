# src/llm/groq_client.py
from groq import Groq, RateLimitError
from src.llm.base_client import BaseLLMClient, LLMRateLimitError
from configs.settings import settings


class GroqClient(BaseLLMClient):
    def __init__(self):
        self.client = Groq(
            api_key=settings.GROQ_API_KEY,
            timeout=settings.LLM_TIMEOUT,
            max_retries=settings.LLM_MAX_RETRIES,
        )
        self.model = settings.GROQ_MODEL

    def chat(
        self, system_prompt: str, user_message: str, temperature: float = 0.1
    ) -> str:
        try:
            return self._chat(system_prompt, user_message, temperature)
        except RateLimitError as exc:
            delay = exc.response.headers.get('retry-after', '30')
            try:
                delay = float(delay)
            except ValueError:
                delay = 30
            raise LLMRateLimitError(delay) from None

    def _chat(self, system_prompt, user_message, temperature):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        )
        return resp.choices[0].message.content
