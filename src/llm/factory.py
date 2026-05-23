# src/llm/factory.py
from configs.settings import settings


def get_llm_client():
    if settings.LLM_PROVIDER == "ollama":
        from src.llm.ollama_client import OllamaClient
        return OllamaClient()
    elif settings.LLM_PROVIDER == "groq":
        from src.llm.groq_client import GroqClient
        return GroqClient()
    elif settings.LLM_PROVIDER == "gemini":
        from src.llm.gemini_client import GeminiClient
        return GeminiClient()
    else:
        raise ValueError(f"Unknown provider: {settings.LLM_PROVIDER}")