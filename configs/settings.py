# configs/settings.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Embedding
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_BATCH_SIZE: int = 32

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_PATH: str = "data/qdrant_storage"
    COLLECTION_NAME: str = "legal_vietnam"
    VECTOR_SIZE: int = 1024

    # Retrieval
    TOP_K_VECTOR: int = 20
    TOP_K_BM25: int = 20
    TOP_K_RERANK: int = 8

    # Reranker
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MIN_SCORE: float = 0.1

    # LLM Provider — đổi ở đây hoặc trong .env
    LLM_PROVIDER: str = "groq"  # "ollama" | "groq" | "gemini"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:7b"

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Paths
    BM25_INDEX_PATH: str = "data/processed/bm25_index.pkl"

    class Config:
        env_file = ".env"


settings = Settings()