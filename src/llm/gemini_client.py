import httpx
from src.llm.base_client import BaseLLMClient
from configs.settings import settings


class GeminiClient(BaseLLMClient):
    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required")
        self.model = settings.GEMINI_MODEL

    def chat(self, system_prompt, user_message, temperature=0.1):
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_message}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": settings.LLM_MAX_OUTPUT_TOKENS,
            },
        }
        with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
            response = client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                headers={"x-goog-api-key": settings.GEMINI_API_KEY},
                json=payload,
            )
            response.raise_for_status()
        return "".join(
            p.get("text", "")
            for p in response.json()["candidates"][0]["content"]["parts"]
        )
