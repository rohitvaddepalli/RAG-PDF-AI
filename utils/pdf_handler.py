import fitz
import os


def save_uploaded_file(uploaded_file, save_dir: str = "uploads") -> str:
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def extract_text_from_pdf(file_path: str) -> dict:
    pages_text = []  
    doc = fitz.open(file_path)
    for page_number in range(len(doc)):
        page = doc[page_number]
        text = page.get_text("text")
        pages_text.append(text)

    doc.close()

    return {
        "pages": pages_text,
        "full_text": "\n\n".join(pages_text),
        "page_count": len(pages_text),
    }


def get_pdf_metadata(file_path: str) -> dict:
    doc = fitz.open(file_path)
    metadata = doc.metadata
    doc.close()
    return metadata