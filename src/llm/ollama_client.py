# src/llm/ollama_client.py
import httpx
from src.llm.base_client import BaseLLMClient
from configs.settings import settings


class OllamaClient(BaseLLMClient):
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.LLM_MODEL

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        stream: bool = False,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "options": {
                "temperature": temperature,
                "num_ctx": settings.LLM_CONTEXT_TOKENS,
                "num_predict": settings.LLM_MAX_OUTPUT_TOKENS,
            },
            "stream": stream,
        }

        with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()

        return resp.json()["message"]["content"]
