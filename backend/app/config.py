"""Configuration settings for the RAG PDF AI backend."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application configuration loaded from environment variables."""

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    EMBEDDING_MODEL_NAME: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.20"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "4"))
    MAX_FILE_SIZE_BYTES: int = 20 * 1024 * 1024

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "650"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))


settings = Settings()
