# src/llm/groq_client.py
from groq import Groq
from src.llm.base_client import BaseLLMClient
from configs.settings import settings


class GroqClient(BaseLLMClient):
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1
    ) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=temperature,
            max_tokens=2048
        )
        return resp.choices[0].message.content