"""Unit tests for pdf_service."""

import pytest
import fitz
from app.services.pdf_service import PDFExtractionError, extract_pages_from_pdf_bytes


def create_sample_pdf_bytes(pages_text: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_extract_pages_success():
    pdf_bytes = create_sample_pdf_bytes([
        "Page 1: Introduction to Retrieval-Augmented Generation.",
        "Page 2: Vector embeddings and cosine similarity retrieval.",
    ])
    pages = extract_pages_from_pdf_bytes(pdf_bytes)
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert "Retrieval-Augmented" in pages[0]["text"]
    assert pages[1]["page_number"] == 2
    assert "Vector embeddings" in pages[1]["text"]


def test_extract_empty_bytes_raises_error():
    with pytest.raises(PDFExtractionError, match="empty"):
        extract_pages_from_pdf_bytes(b"")


def test_extract_corrupted_pdf_raises_error():
    with pytest.raises(PDFExtractionError, match="Corrupted or invalid"):
        extract_pages_from_pdf_bytes(b"Not a valid PDF file content")


def test_scanned_zero_text_pdf_raises_error():
    doc = fitz.open()
    doc.new_page()
    pdf_bytes = doc.write()
    doc.close()

    with pytest.raises(PDFExtractionError, match="Scanned or image-only"):
        extract_pages_from_pdf_bytes(pdf_bytes)
