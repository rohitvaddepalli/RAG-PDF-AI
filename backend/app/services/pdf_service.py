"""Service for extracting and validating text from PDF documents using PyMuPDF."""

from typing import Any, Dict, List
import fitz  # PyMuPDF


class PDFExtractionError(Exception):
    """Raised when PDF extraction fails due to formatting, corruption, or scanning."""
    pass


def extract_pages_from_pdf_bytes(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extract text page by page from PDF raw bytes.

    Args:
        pdf_bytes: Raw binary content of the PDF file.

    Returns:
        List of dicts: [{"page_number": int, "text": str}]

    Raises:
        PDFExtractionError: If file is empty, corrupted, encrypted, or has no extractable text.
    """
    if not pdf_bytes or len(pdf_bytes) == 0:
        raise PDFExtractionError("The uploaded file is empty (0 bytes).")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise PDFExtractionError(f"Corrupted or invalid PDF file: {str(e)}")

    if doc.is_encrypted:
        doc.close()
        raise PDFExtractionError("The uploaded PDF is password-protected and cannot be read.")

    total_pages = len(doc)
    if total_pages == 0:
        doc.close()
        raise PDFExtractionError("The PDF contains 0 pages.")

    pages: List[Dict[str, Any]] = []
    total_words = 0

    try:
        for page_idx, page in enumerate(doc, start=1):
            text = page.get_text("text", sort=True) or ""
            cleaned = text.strip()
            total_words += len(cleaned.split())
            pages.append({
                "page_number": page_idx,
                "text": cleaned,
            })
    finally:
        doc.close()

    # Detect scanned or image-only documents
    if total_words == 0:
        raise PDFExtractionError(
            "Scanned or image-only PDF: No extractable text found. "
            "Please upload a document with selectable text."
        )

    return pages
