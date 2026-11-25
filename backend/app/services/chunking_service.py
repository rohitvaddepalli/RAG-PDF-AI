"""Service for splitting extracted page text into structured overlapping chunks."""

from typing import Any, Dict, List
import re

DEFAULT_CHUNK_SIZE: int = 650
DEFAULT_CHUNK_OVERLAP: int = 100


def split_text_recursive(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a block of text into chunks of approx chunk_size characters with overlap.
    Splits naturally along paragraph breaks, sentence breaks, spaces, or character boundaries.
    """
    if not text or not text.strip():
        return []

    cleaned_text = re.sub(r"[ \t]+", " ", text.strip())
    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks: List[str] = []
    start = 0
    text_len = len(cleaned_text)

    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(cleaned_text[start:].strip())
            break

        slice_window = cleaned_text[start:end]
        split_pos = -1

        for delimiter in ["\n\n", "\n", ". ", "? ", "! ", "; ", " "]:
            last_idx = slice_window.rfind(delimiter)
            if last_idx > int(chunk_size * 0.4):
                split_pos = last_idx + len(delimiter)
                break

        if split_pos != -1:
            chunk = cleaned_text[start : start + split_pos].strip()
            if chunk:
                chunks.append(chunk)
            start = start + split_pos - chunk_overlap
        else:
            chunks.append(cleaned_text[start:end].strip())
            start = end - chunk_overlap

        if start <= 0 or start >= text_len:
            break

    return [c for c in chunks if c]


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Chunk extracted pages, tagging every chunk with its 1-indexed page number.

    Args:
        pages: List of dicts with 'page_number' and 'text'.
        chunk_size: Target size in characters (approx 500-800).
        chunk_overlap: Number of characters to overlap between adjacent chunks.

    Returns:
        List of dicts: [
            {
                "chunk_id": str,
                "page": int,
                "text": str,
                "char_count": int,
            }
        ]
    """
    all_chunks: List[Dict[str, Any]] = []

    for page_dict in pages:
        page_num = page_dict.get("page_number", 1)
        raw_text = page_dict.get("text", "")
        if not raw_text or not raw_text.strip():
            continue

        page_chunks = split_text_recursive(
            raw_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for idx, chunk_text in enumerate(page_chunks):
            all_chunks.append({
                "chunk_id": f"p{page_num}_c{idx}",
                "page": page_num,
                "text": chunk_text,
                "char_count": len(chunk_text),
            })

    return all_chunks
