"""Unit tests for chunking_service."""

from app.services.chunking_service import chunk_pages, split_text_recursive


def test_split_text_small():
    short_text = "This is a short paragraph that easily fits within 500 characters."
    chunks = split_text_recursive(short_text, chunk_size=650)
    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_split_text_long_with_overlap():
    long_text = "Sentence number one. " * 50
    chunks = split_text_recursive(long_text, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= 600


def test_chunk_pages_preserves_page_numbers():
    pages = [
        {"page_number": 1, "text": "Content on page one. " * 20},
        {"page_number": 2, "text": "Content on page two. " * 30},
    ]
    chunks = chunk_pages(pages, chunk_size=500, chunk_overlap=100)
    assert len(chunks) >= 2
    page_numbers = {c["page"] for c in chunks}
    assert page_numbers == {1, 2}
    assert all("chunk_id" in c for c in chunks)
    assert all("char_count" in c for c in chunks)
