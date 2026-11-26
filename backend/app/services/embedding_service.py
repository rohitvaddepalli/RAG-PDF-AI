"""Service for generating local dense embeddings using SentenceTransformers."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def load_model(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not self._model:
            self.load_model()

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        if not self._model:
            self.load_model()

        query_cleaned = query.strip()
        vector = self._model.encode(
            query_cleaned,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vector.astype(np.float32).flatten()


embedding_service = EmbeddingService()
