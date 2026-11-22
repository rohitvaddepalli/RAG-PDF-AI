import json
import math
import os
from typing import Any, Dict, List, Optional
import numpy as np

from utils.embeddings import EMBEDDING_DIMENSION


class VectorStore:
    def __init__(
        self,
        expected_dim: int = EMBEDDING_DIMENSION,
        file_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        self.expected_dim = expected_dim
        self.file_path = file_path or (db_path if db_path != ":memory:" else None)
        self.chunks: List[Dict[str, Any]] = []
        self._matrix: Optional[np.ndarray] = None

    def _rebuild_matrix(self) -> None:
        if not self.chunks:
            self._matrix = None
            return
        vectors = [chunk["embedding"] for chunk in self.chunks]
        self._matrix = np.array(vectors, dtype=np.float32)

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        upsert: bool = True,
    ) -> int:
        if not chunks:
            return 0

        added_count = 0
        chunk_map = {c["chunk_id"]: idx for idx, c in enumerate(self.chunks) if "chunk_id" in c}

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk at index {idx} must be a dictionary.")

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            embedding = chunk.get("embedding")

            if not text or not isinstance(text, str):
                raise ValueError(f"Chunk at index {idx} has invalid or missing 'text'.")
            if embedding is None or not isinstance(embedding, (list, tuple, np.ndarray)):
                raise ValueError(f"Chunk at index {idx} has invalid or missing 'embedding'.")

            if len(embedding) != self.expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.expected_dim}, got {len(embedding)}."
                )

            emb_list = [float(v) for v in embedding]
            for dim_idx, val in enumerate(emb_list):
                if not math.isfinite(val):
                    raise ValueError(
                        f"Non-finite float value ({val}) at dimension {dim_idx}."
                    )

            cleaned_chunk = dict(chunk)
            cleaned_chunk["embedding"] = emb_list

            if chunk_id and chunk_id in chunk_map:
                if upsert:
                    existing_idx = chunk_map[chunk_id]
                    self.chunks[existing_idx] = cleaned_chunk
                    added_count += 1
                else:
                    raise ValueError(f"Duplicate chunk_id '{chunk_id}' already exists.")
            else:
                if chunk_id:
                    chunk_map[chunk_id] = len(self.chunks)
                self.chunks.append(cleaned_chunk)
                added_count += 1

        self._rebuild_matrix()
        return added_count

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 3,
        score_threshold: Optional[float] = None,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if query_embedding is None or len(query_embedding) == 0:
            raise ValueError("Query embedding cannot be empty or None.")

        if len(query_embedding) != self.expected_dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.expected_dim}, "
                f"got {len(query_embedding)}."
            )

        query_arr = np.array(query_embedding, dtype=np.float32)
        if not np.all(np.isfinite(query_arr)):
            raise ValueError("Query embedding contains non-finite values (NaN or Inf).")

        query_norm = float(np.linalg.norm(query_arr))
        if query_norm == 0.0 or math.isclose(query_norm, 0.0, abs_tol=1e-9):
            raise ValueError("Query embedding cannot be a zero vector (norm is 0).")

        if not self.chunks or top_k <= 0 or self._matrix is None:
            return []

        filtered_indices = []
        for idx, chunk in enumerate(self.chunks):
            if source_filter and chunk.get("source") != source_filter:
                continue
            filtered_indices.append(idx)

        if not filtered_indices:
            return []

        matrix = self._matrix[filtered_indices]
        chunk_norms = np.linalg.norm(matrix, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            dots = np.dot(matrix, query_arr)
            denominators = chunk_norms * query_norm
            similarities = np.where(denominators > 1e-9, dots / denominators, 0.0)
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)
            similarities = np.clip(similarities, -1.0, 1.0)

        sorted_order = np.argsort(-similarities)

        results: List[Dict[str, Any]] = []
        for pos in sorted_order:
            score = float(similarities[pos])
            if score_threshold is not None and score < score_threshold:
                continue

            orig_idx = filtered_indices[pos]
            chunk_copy = dict(self.chunks[orig_idx])
            chunk_copy["similarity"] = score
            chunk_copy["score"] = score
            results.append(chunk_copy)

            if len(results) >= top_k:
                break

        return results

    def save(self, filepath: Optional[str] = None) -> None:
        target_path = filepath or self.file_path
        if not target_path:
            raise ValueError("No file path specified for saving.")

        dir_name = os.path.dirname(target_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

    def load(self, filepath: Optional[str] = None) -> None:
        target_path = filepath or self.file_path
        if not target_path:
            raise ValueError("No file path specified for loading.")

        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Vector store file not found: {target_path}")

        with open(target_path, "r", encoding="utf-8") as f:
            loaded_chunks = json.load(f)

        if not isinstance(loaded_chunks, list):
            raise ValueError("Invalid format: expected a JSON list of chunks.")

        self.chunks = []
        self.add_chunks(loaded_chunks)

    def count(self, source_filter: Optional[str] = None) -> int:
        if source_filter:
            return sum(1 for c in self.chunks if c.get("source") == source_filter)
        return len(self.chunks)

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        for chunk in self.chunks:
            if chunk.get("chunk_id") == chunk_id:
                return dict(chunk)
        return None

    def delete_chunk(self, chunk_id: str) -> bool:
        for idx, chunk in enumerate(self.chunks):
            if chunk.get("chunk_id") == chunk_id:
                del self.chunks[idx]
                self._rebuild_matrix()
                return True
        return False

    def clear(self) -> None:
        self.chunks = []
        self._matrix = None

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
