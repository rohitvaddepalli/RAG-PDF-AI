from typing import Any, Dict, List, Optional
import numpy as np

from app.config import settings


class DocumentStore:
    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def add_document(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        page_count: int,
        file_name: Optional[str] = None,
    ) -> None:
        self._store[document_id] = {
            "chunks": chunks,
            "embeddings": embeddings,
            "page_count": page_count,
            "file_name": file_name or "document.pdf",
        }

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(document_id)

    def delete_document(self, document_id: str) -> bool:
        if document_id in self._store:
            del self._store[document_id]
            return True
        return False

    def document_count(self) -> int:
        return len(self._store)

    def search(
        self,
        document_id: str,
        query_vector: np.ndarray,
        top_k: int = settings.DEFAULT_TOP_K,
        threshold: float = settings.RELEVANCE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Search for top_k relevant chunks within a document using NumPy cosine similarity.
        Chunks with similarity score strictly below threshold are excluded.

        Args:
            document_id: Target document ID.
            query_vector: 1D normalized NumPy vector (dim 384).
            top_k: Max number of results to return.
            threshold: Minimum cosine similarity threshold.

        Returns:
            List of matching chunks sorted by relevance:
            [{"page": int, "snippet": str, "score": float, "chunk_id": str}]
        """
        doc = self.get_document(document_id)
        if not doc:
            raise KeyError(f"Document with ID '{document_id}' not found.")

        chunks = doc["chunks"]
        matrix = doc["embeddings"]

        if len(chunks) == 0 or matrix.shape[0] == 0:
            return []

        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []

        matrix_norms = np.linalg.norm(matrix, axis=1)
        denominators = matrix_norms * query_norm
        denominators = np.where(denominators > 1e-9, denominators, 1e-9)

        dots = np.dot(matrix, query_vector)
        similarities = dots / denominators
        similarities = np.clip(similarities, -1.0, 1.0)

        sorted_indices = np.argsort(-similarities)

        results: List[Dict[str, Any]] = []
        for idx in sorted_indices:
            score = float(similarities[idx])
            if score < threshold:
                break

            chunk_info = chunks[idx]
            results.append({
                "page": chunk_info["page"],
                "snippet": chunk_info["text"],
                "score": round(score, 4),
                "chunk_id": chunk_info.get("chunk_id", ""),
            })

            if len(results) >= top_k:
                break

        return results


document_store = DocumentStore()