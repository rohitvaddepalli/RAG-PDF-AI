import pytest
from app import index_pdf_document
from utils.vector_store import VectorStore

def test_index_pdf_document_creates_populated_vector_store():
    fake_pdf_data = {
        "file_path": "uploads/fake.pdf",
        "file_name": "fake.pdf",
        "pages": [
            {
                "page_number": 1,
                "text": "Artificial Intelligence is transforming search and retrieval applications across all domains.",
            },
            {
                "page_number": 2,
                "text": "Vector databases enable semantic similarity search using dense numerical embeddings.",
            },
        ],
    }
    store = index_pdf_document(fake_pdf_data)
    assert isinstance(store, VectorStore)
    assert store.count() >= 2
    assert store.chunks[0]["page_number"] == 1
    assert "embedding" in store.chunks[0]
    assert len(store.chunks[0]["embedding"]) == 384

def test_index_pdf_document_empty_pages_returns_empty_store():
    empty_pdf_data = {
        "file_path": "uploads/empty.pdf",
        "file_name": "empty.pdf",
        "pages": [],
    }
    store = index_pdf_document(empty_pdf_data)
    assert isinstance(store, VectorStore)
    assert store.count() == 0

def test_index_pdf_document_blank_text_pages():
    blank_pdf_data = {
        "file_path": "uploads/blank.pdf",
        "file_name": "blank.pdf",
        "pages": [
            {"page_number": 1, "text": "   "},
            {"page_number": 2, "text": ""},
        ],
    }
    store = index_pdf_document(blank_pdf_data)
    assert isinstance(store, VectorStore)
    assert store.count() == 0
