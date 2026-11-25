"""Service for generating local dense embeddings using SentenceTransformers."""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """Manages the SentenceTransformer model loaded in memory."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def load_model(self) -> None:
        """Load the model into memory. Called once during application startup."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

    @property
    def is_loaded(self) -> bool:
        """Check if the embedding model is loaded."""
        return self._model is not None

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a list of text strings into a 2D NumPy array of embeddings.

        Args:
            texts: List of strings to encode.
            batch_size: Batch size for encoding.

        Returns:
            np.ndarray of shape (len(texts), 384) with float32 values.
        """
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
        """
        Encode a single question string into a normalized 1D NumPy vector.

        Args:
            query: Question text.

        Returns:
            np.ndarray of shape (384,) with float32 values.
        """
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
