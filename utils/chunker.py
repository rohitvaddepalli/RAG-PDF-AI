import os
import re
from typing import Any, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE: int = 700
DEFAULT_CHUNK_OVERLAP: int = 100


def get_text_splitter(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )


def generate_chunk_id(source_name: str, page_number: int, chunk_index: int) -> str:
    clean_filename = os.path.basename(source_name).strip()
    clean_filename = re.sub(r"[^\w.\-]", "_", clean_filename)
    return f"{clean_filename}_p{page_number}_c{chunk_index}"


def chunk_pages(
    pages: List[Dict[str, Any]],
    source_name: str = "document.pdf",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    if not pages:
        return []

    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    structured_chunks: List[Dict[str, Any]] = []

    for page in pages:
        raw_text = page.get("text", "")
        if not raw_text or not raw_text.strip():
            continue

        page_number = page.get("page_number", 1)
        text_content = raw_text.strip()

        raw_chunks = splitter.split_text(text_content)

        for chunk_idx, chunk_text in enumerate(raw_chunks):
            cleaned_chunk = chunk_text.strip()
            if not cleaned_chunk:
                continue

            chunk_id = generate_chunk_id(
                source_name=source_name,
                page_number=page_number,
                chunk_index=chunk_idx,
            )

            structured_chunks.append({
                "chunk_id": chunk_id,
                "text": cleaned_chunk,
                "page_number": page_number,
                "source": source_name,
                "chunk_index": chunk_idx,
                "word_count": len(cleaned_chunk.split()),
                "char_count": len(cleaned_chunk),
            })

    return structured_chunks


def chunk_extracted_pdf(
    pdf_data: Dict[str, Any],
    source_name: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    if not pdf_data:
        return []

    resolved_source = source_name or pdf_data.get("file_name", "document.pdf")
    pages = pdf_data.get("pages", [])

    return chunk_pages(
        pages=pages,
        source_name=resolved_source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
