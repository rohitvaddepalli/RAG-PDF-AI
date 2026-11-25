"""Unit tests for retrieval_service."""

import numpy as np
import pytest
from app.services.retrieval_service import DocumentStore


def test_document_store_add_and_search():
    store = DocumentStore()
    doc_id = "test_doc_1"
    chunks = [
        {"page": 1, "text": "Python is a popular programming language.", "chunk_id": "c1"},
        {"page": 2, "text": "Photosynthesis is the process by which plants make food.", "chunk_id": "c2"},
    ]
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    matrix = np.vstack([emb1, emb2])

    store.add_document(doc_id, chunks, matrix, page_count=2)
    assert store.document_count() == 1

    query_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    results = store.search(doc_id, query_vec, top_k=2, threshold=0.3)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    assert results[0]["page"] == 1
    assert results[0]["score"] == 1.0


def test_document_store_threshold_filtering():
    store = DocumentStore()
    doc_id = "test_doc_threshold"
    chunks = [{"page": 1, "text": "Topic A", "chunk_id": "c1"}]
    emb = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    matrix = np.vstack([emb])
    store.add_document(doc_id, chunks, matrix, page_count=1)

    query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    results = store.search(doc_id, query, top_k=4, threshold=0.3)
    assert len(results) == 0


def test_document_store_missing_doc_raises_keyerror():
    store = DocumentStore()
    with pytest.raises(KeyError, match="not found"):
        store.search("non_existent_id", np.array([1.0, 0.0]))
