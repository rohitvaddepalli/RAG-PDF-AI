from functools import lru_cache
import math
from typing import Any, Dict, List, Optional
import numpy as np

DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384
DEFAULT_BATCH_SIZE: int = 32


@lru_cache(maxsize=1)
def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


get_embedding_model = load_embedding_model


def validate_embeddings(
    embeddings: List[List[float]],
    expected_count: int,
    expected_dim: int = EMBEDDING_DIMENSION,
) -> None:
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


def embed_query(
    query: str,
    model: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[float]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query text cannot be empty or non-string.")

    active_model = model if model is not None else load_embedding_model(model_name)

    raw_vector = active_model.encode(query.strip(), show_progress_bar=False)

    if hasattr(raw_vector, "tolist"):
        vector: List[float] = raw_vector.tolist()
    else:
        vector = [float(val) for val in raw_vector]

    validate_embeddings(
        embeddings=[vector],
        expected_count=1,
        expected_dim=EMBEDDING_DIMENSION,
    )

    return vector
