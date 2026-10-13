# configs/settings.py
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")
    # Embedding
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_BATCH_SIZE: int = 32
    MODEL_DEVICE: str = "cpu"

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_PATH: str = "data/qdrant_storage"
    COLLECTION_NAME: str = "legal_vietnam"
    VECTOR_SIZE: int = 1024
    QDRANT_MODE: str = "local"
    ACTIVE_INDEX_PATH: str = "data/indexes/active.json"
    QDRANT_TIMEOUT: float = 15

    # Retrieval
    TOP_K_VECTOR: int = 20
    TOP_K_BM25: int = 20
    TOP_K_RERANK: int = 8
    RETRIEVAL_CANDIDATES: int = 40
    RRF_K: int = 60

    # Query Rewriting
    ENABLE_QUERY_REWRITE: bool = True
    ENABLE_QUERY_ALIASES: bool = True

    # Reranker
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MIN_SCORE: float = (
        -10.0
    )  # Provisional raw logit, not a confidence probability.
    RERANK_MAX_PER_ARTICLE: int = 0
    CONTEXT_TOKEN_BUDGET: int = 9000
    LLM_CONTEXT_TOKENS: int = 16384
    LLM_MAX_OUTPUT_TOKENS: int = 1800
    LLM_TIMEOUT: float = 60
    LLM_MAX_RETRIES: int = 1
    MAX_QUERY_CHARS: int = 3000
    MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024
    MAX_DOCUMENT_CHARS: int = 150000
    MAX_DOCUMENT_SECTIONS: int = 30
    DOCUMENT_SECTION_CHARS: int = 2500
    MAX_CONCURRENT_REQUESTS: int = 1
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])

    LLM_PROVIDER: str = "groq"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:7b"

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "qwen/qwen3.8-27b"

    # Gemini
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Paths
    BM25_INDEX_PATH: str = "data/processed/bm25_index.pkl"


settings = Settings()


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path
