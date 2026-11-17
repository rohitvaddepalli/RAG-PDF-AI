import json
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional
import numpy as np

from utils.embeddings import EMBEDDING_DIMENSION


class VectorStore:
    def __init__(
        self,
        db_path: str = "vector_store.db",
        expected_dim: int = EMBEDDING_DIMENSION,
    ):
        self.db_path = db_path
        self.expected_dim = expected_dim

        if self.db_path == ":memory:":
            self._memory_conn: Optional[sqlite3.Connection] = sqlite3.connect(
                ":memory:", check_same_thread=False
            )
            self._memory_conn.row_factory = sqlite3.Row
        else:
            self._memory_conn = None
            dir_name = os.path.dirname(self.db_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn

        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                        chunk_id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page_number INTEGER NOT NULL,
                        chunk_index INTEGER,
                        word_count INTEGER,
                        char_count INTEGER,
                        embedding TEXT NOT NULL
                    )
                    """
                )
                # Create an index on source to accelerate per-document queries and cleanup
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);"
                )
        finally:
            if self._memory_conn is None:
                conn.close()

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        upsert: bool = True,
    ) -> int:
        if not chunks:
            return 0

        prepared_records = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValueError(f"Chunk at index {idx} must be a dictionary.")

            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text")
            source = chunk.get("source")
            page_number = chunk.get("page_number")
            embedding = chunk.get("embedding")

            if not chunk_id or not isinstance(chunk_id, str):
                raise ValueError(f"Chunk at index {idx} has an invalid or missing 'chunk_id'.")
            if not text or not isinstance(text, str):
                raise ValueError(f"Chunk '{chunk_id}' has invalid or missing 'text'.")
            if not source or not isinstance(source, str):
                raise ValueError(f"Chunk '{chunk_id}' has invalid or missing 'source'.")
            if page_number is None or not isinstance(page_number, int):
                raise ValueError(f"Chunk '{chunk_id}' has invalid or missing 'page_number'.")
            if embedding is None or not isinstance(embedding, (list, tuple, np.ndarray)):
                raise ValueError(f"Chunk '{chunk_id}' has invalid or missing 'embedding'.")

            if len(embedding) != self.expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch in chunk '{chunk_id}': "
                    f"expected {self.expected_dim}, got {len(embedding)}."
                )

            for dim_idx, val in enumerate(embedding):
                if not isinstance(val, (int, float, np.floating, np.integer)) or not math.isfinite(val):
                    raise ValueError(
                        f"Non-finite float value ({val}) in chunk '{chunk_id}' at dimension {dim_idx}."
                    )

            chunk_index = chunk.get("chunk_index", 0)
            word_count = chunk.get("word_count", len(text.split()))
            char_count = chunk.get("char_count", len(text))
            embedding_json = json.dumps([float(v) for v in embedding])

            prepared_records.append((
                chunk_id,
                text,
                source,
                page_number,
                chunk_index,
                word_count,
                char_count,
                embedding_json,
            ))

        sql = (
            """
            INSERT INTO chunks (
                chunk_id, text, source, page_number, chunk_index, word_count, char_count, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id) DO UPDATE SET
                text = excluded.text,
                source = excluded.source,
                page_number = excluded.page_number,
                chunk_index = excluded.chunk_index,
                word_count = excluded.word_count,
                char_count = excluded.char_count,
                embedding = excluded.embedding
            """
            if upsert
            else """
            INSERT INTO chunks (
                chunk_id, text, source, page_number, chunk_index, word_count, char_count, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
        )

        conn = self._get_connection()
        try:
            with conn:
                conn.executemany(sql, prepared_records)
            return len(prepared_records)
        finally:
            if self._memory_conn is None:
                conn.close()

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
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

        if top_k <= 0:
            return []

        conn = self._get_connection()
        try:
            if source_filter:
                cursor = conn.execute(
                    """
                    SELECT chunk_id, text, source, page_number, chunk_index, word_count, char_count, embedding
                    FROM chunks WHERE source = ?
                    """,
                    (source_filter,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT chunk_id, text, source, page_number, chunk_index, word_count, char_count, embedding
                    FROM chunks
                    """
                )
            rows = cursor.fetchall()
        finally:
            if self._memory_conn is None:
                conn.close()

        if not rows:
            return []

        matrix = np.array([json.loads(row["embedding"]) for row in rows], dtype=np.float32)

        chunk_norms = np.linalg.norm(matrix, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            similarities = np.dot(matrix, query_arr) / (chunk_norms * query_norm)
            similarities = np.nan_to_num(similarities, nan=0.0, posinf=0.0, neginf=0.0)

        similarities = np.clip(similarities, -1.0, 1.0)

        sorted_indices = np.argsort(-similarities)

        results: List[Dict[str, Any]] = []
        for idx in sorted_indices:
            score = float(similarities[idx])
            if score_threshold is not None and score < score_threshold:
                continue

            row = rows[idx]
            results.append({
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "source": row["source"],
                "page_number": row["page_number"],
                "chunk_index": row["chunk_index"],
                "word_count": row["word_count"],
                "char_count": row["char_count"],
                "similarity": score,
                "score": score,
            })

            if len(results) >= top_k:
                break

        return results

    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                """
                SELECT chunk_id, text, source, page_number, chunk_index, word_count, char_count, embedding
                FROM chunks WHERE chunk_id = ?
                """,
                (chunk_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            return {
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "source": row["source"],
                "page_number": row["page_number"],
                "chunk_index": row["chunk_index"],
                "word_count": row["word_count"],
                "char_count": row["char_count"],
                "embedding": json.loads(row["embedding"]),
            }
        finally:
            if self._memory_conn is None:
                conn.close()

    def delete_chunk(self, chunk_id: str) -> bool:
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.execute("DELETE FROM chunks WHERE chunk_id = ?", (chunk_id,))
                return cursor.rowcount > 0
        finally:
            if self._memory_conn is None:
                conn.close()

    def count(self, source_filter: Optional[str] = None) -> int:
        conn = self._get_connection()
        try:
            if source_filter:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE source = ?", (source_filter,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            return cursor.fetchone()[0]
        finally:
            if self._memory_conn is None:
                conn.close()

    def clear(self) -> None:
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM chunks")
        finally:
            if self._memory_conn is None:
                conn.close()

    def close(self) -> None:
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
