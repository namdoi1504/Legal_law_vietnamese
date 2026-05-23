# src/llm/base_client.py
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1
    ) -> str:
        pass