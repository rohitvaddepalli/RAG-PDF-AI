import os
from typing import Any, Dict, List
import fitz
import uuid


def save_uploaded_file(uploaded_file, save_dir: str = "uploads") -> str:
    os.makedirs(save_dir, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(save_dir, safe_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def extract_text_from_pdf(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        return {
            "status": "error",
            "error_message": f"File not found: {file_path}",
            "page_count": 0,
            "total_words": 0,
            "has_extractable_text": False,
            "pages": [],
            "full_text": "",
        }

    if os.path.getsize(file_path) == 0:
        return {
            "status": "error",
            "error_message": "The selected PDF file is completely empty (0 bytes).",
            "page_count": 0,
            "total_words": 0,
            "has_extractable_text": False,
            "pages": [],
            "full_text": "",
        }

    try:
        pages: List[Dict[str, Any]] = []
        total_words = 0

        with fitz.open(file_path) as doc:
            page_count = len(doc)

            if page_count == 0:
                return {
                    "status": "error",
                    "error_message": "The PDF contains 0 pages.",
                    "page_count": 0,
                    "total_words": 0,
                    "has_extractable_text": False,
                    "pages": [],
                    "full_text": "",
                }

            for page_number, page in enumerate(doc, start=1):
                raw_text = page.get_text("text", sort=True) or ""
                cleaned_text = raw_text.strip()
                words = cleaned_text.split()
                page_words = len(words)
                total_words += page_words

                pages.append({
                    "page_number": page_number,
                    "text": cleaned_text,
                    "word_count": page_words,
                    "char_count": len(cleaned_text),
                })

        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        has_extractable_text = total_words > 0

        return {
            "status": "success",
            "error_message": None,
            "page_count": page_count,
            "total_words": total_words,
            "has_extractable_text": has_extractable_text,
            "pages": pages,
            "full_text": full_text,
        }

    except (fitz.FileDataError, fitz.EmptyFileError) as e:
        return {
            "status": "error",
            "error_message": f"Corrupted or invalid PDF format: {str(e)}",
            "page_count": 0,
            "total_words": 0,
            "has_extractable_text": False,
            "pages": [],
            "full_text": "",
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Unexpected error while reading PDF: {str(e)}",
            "page_count": 0,
            "total_words": 0,
            "has_extractable_text": False,
            "pages": [],
            "full_text": "",
        }


def get_pdf_metadata(file_path: str) -> dict:
    try:
        with fitz.open(file_path) as doc:
            return doc.metadata or {}
    except Exception:
        return {}
