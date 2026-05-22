from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Embedding
    EMBED_MODEL: str = "BAAI/bge-m3"
    EMBED_BATCH_SIZE: int = 32

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    COLLECTION_NAME: str = "legal_vietnam"
    VECTOR_SIZE: int = 1024  # bge-m3 output dimension

    # Retrieval
    TOP_K_VECTOR: int = 10
    TOP_K_BM25: int = 10
    TOP_K_RERANK: int = 5

    # Reranker
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_MIN_SCORE: float = 0.1  # Ngưỡng hallucination guard

    # LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen2.5:7b"
    LLM_TEMPERATURE: float = 0.1  # Thấp = ít hallucinate

    # Paths
    BM25_INDEX_PATH: str = "data/processed/bm25_index.pkl"
    PROCESSED_DATA_DIR: str = "data/processed"
    RAW_DATA_DIR: str = "data/raw"

    class Config:
        env_file = ".env"


settings = Settings()