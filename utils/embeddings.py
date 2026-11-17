"""
utils/embeddings.py
-------------------
Module for generating vector embeddings from text chunks using Sentence Transformers.

Key Features:
- Uses 'sentence-transformers/all-MiniLM-L6-v2' (producing 384-dimensional dense vectors).
- Caches model loading in memory so weights are only loaded once per session.
- Batched encoding for fast, memory-efficient processing.
- Preserves all chunk metadata from utils/chunker.py and adds 'embedding' as a Python list of floats.
- Validates embedding count, dimensionality (384), and finite numerical values (no NaN/Inf).
- Supports dependency injection of mock models for instantaneous, offline unit testing.
- Safe handling of empty inputs, non-dict items, and missing/empty text fields.
"""

from functools import lru_cache
import math
from typing import Any, Dict, List, Optional
import numpy as np

DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384
DEFAULT_BATCH_SIZE: int = 32


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """
    Loads and caches the SentenceTransformer model instance in memory.

    Using Python's built-in @lru_cache ensures the neural network weights
    are loaded exactly once per application lifecycle, saving RAM and
    eliminating redundant initialization overhead.

    Args:
        model_name: Hugging Face repository ID or local model path.
                    Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        SentenceTransformer: The initialized embedding model.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


get_embedding_model = load_embedding_model


def validate_embeddings(
    embeddings: List[List[float]],
    expected_count: int,
    expected_dim: int = EMBEDDING_DIMENSION,
) -> None:
    """
    Validates that a batch of generated embeddings meets integrity requirements.

    Checks:
    1. Count match: Number of embedding vectors matches number of chunks.
    2. Dimension match: Each vector matches expected_dim (384).
    3. Value sanity: No vector contains NaN (Not a Number) or Infinite (+/-Inf) values.

    Args:
        embeddings: List of embedding vectors (each vector is a list of floats).
        expected_count: Number of chunks expected.
        expected_dim: Dimensionality expected for each vector (default: 384).

    Raises:
        ValueError: If any check fails.
    """
    if len(embeddings) != expected_count:
        raise ValueError(
            f"Embedding count mismatch: expected {expected_count} vectors, "
            f"but received {len(embeddings)}."
        )

    for idx, vector in enumerate(embeddings):
        if not isinstance(vector, (list, tuple, np.ndarray)):
            raise ValueError(
                f"Vector at index {idx} must be a list or array, got {type(vector).__name__}."
            )

        if len(vector) != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch at chunk index {idx}: "
                f"expected {expected_dim} dimensions, got {len(vector)}."
            )

        for dim_idx, val in enumerate(vector):
            if not isinstance(val, (int, float, np.floating, np.integer)) or not math.isfinite(val):
                raise ValueError(
                    f"Invalid non-finite float value ({val}) detected at "
                    f"chunk index {idx}, dimension {dim_idx}."
                )


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model: Optional[Any] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[Dict[str, Any]]:
    """
    Generates 384-dimensional vector embeddings for a list of text chunk dictionaries.

    Preserves all original chunk metadata (e.g. chunk_id, page_number, source,
    word_count, char_count) and appends an 'embedding' field containing a native
    Python list of floats.

    Args:
        chunks: List of chunk dictionaries produced by utils.chunker, each containing:
                - 'chunk_id': unique identifier
                - 'text': text content to embed
                - 'page_number', 'source', 'chunk_index', etc.
        model: Optional pre-loaded or mock SentenceTransformer instance.
               If None, automatically loads and caches DEFAULT_MODEL_NAME.
               This parameter enables unit testing without downloading weights.
        batch_size: Number of texts encoded per forward pass (default: 32).
        model_name: Model identifier used if 'model' is not provided.

    Returns:
        List[Dict[str, Any]]: A new list of chunk dictionaries, each augmented with
                              an 'embedding' key: List[float] of length 384.

    Raises:
        ValueError: If generated embeddings fail dimensionality or sanity validation.
    """
    if not chunks:
        return []

    valid_chunks: List[Dict[str, Any]] = []
    texts_to_embed: List[str] = []

    for item in chunks:
        if not isinstance(item, dict):
            continue

        raw_text = item.get("text")
        if isinstance(raw_text, str) and raw_text.strip():
            valid_chunks.append(dict(item))
            texts_to_embed.append(raw_text.strip())

    if not valid_chunks:
        return []

    active_model = model if model is not None else load_embedding_model(model_name)

    raw_embeddings = active_model.encode(
        texts_to_embed,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    if hasattr(raw_embeddings, "tolist"):
        vector_list: List[List[float]] = raw_embeddings.tolist()
    else:
        vector_list = [
            [float(val) for val in vec]
            for vec in raw_embeddings
        ]

    validate_embeddings(
        embeddings=vector_list,
        expected_count=len(valid_chunks),
        expected_dim=EMBEDDING_DIMENSION,
    )

    for chunk_dict, vector in zip(valid_chunks, vector_list):
        chunk_dict["embedding"] = [float(val) for val in vector]

    return valid_chunks
