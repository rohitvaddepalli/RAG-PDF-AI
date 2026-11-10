"""
utils/pdf_handler.py
--------------------
Responsible for everything related to reading and processing PDF files.
We keep this in a separate file so the main app stays clean and readable.
PyMuPDF is imported as `fitz` — that is simply the name of its Python module.
"""

import fitz  # PyMuPDF  ← pip install PyMuPDF installs this
import os


def save_uploaded_file(uploaded_file, save_dir: str = "uploads") -> str:
    """
    Streamlit gives us an UploadedFile object (like an in-memory file).
    We write its raw bytes to disk so PyMuPDF can open it by path later.

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The object returned by st.file_uploader().
    save_dir : str
        Folder where the PDF will be saved on disk.

    Returns
    -------
    str
        The full file path of the saved PDF.
    """
    # Make sure the destination folder exists
    os.makedirs(save_dir, exist_ok=True)

    # Build the full path, e.g. "uploads/my_report.pdf"
    file_path = os.path.join(save_dir, uploaded_file.name)

    # Write the bytes to disk
    # uploaded_file.getbuffer() returns a memoryview of the raw bytes
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def extract_text_from_pdf(file_path: str) -> dict:
    """
    Open a PDF and extract plain text from every page using PyMuPDF.

    Parameters
    ----------
    file_path : str
        Path to the PDF saved on disk.

    Returns
    -------
    dict with keys:
        "pages"       → list of str, one entry per page
        "full_text"   → str, all pages joined together
        "page_count"  → int, total number of pages
    """
    pages_text = []  # We'll store each page's text here

    # fitz.open() loads the PDF into memory
    doc = fitz.open(file_path)

    # Iterate through every page (0-indexed)
    for page_number in range(len(doc)):
        page = doc[page_number]                  # Get the page object
        text = page.get_text("text")             # Extract plain text
        pages_text.append(text)

    doc.close()  # Always close the document to free memory

    return {
        "pages": pages_text,
        "full_text": "\n\n".join(pages_text),    # Combine all pages
        "page_count": len(pages_text),
    }


def get_pdf_metadata(file_path: str) -> dict:
    """
    Read the metadata stored inside the PDF file itself.
    Most PDFs have author, title, creation date, etc. embedded in them.

    Parameters
    ----------
    file_path : str
        Path to the PDF saved on disk.

    Returns
    -------
    dict
        Raw metadata dictionary from PyMuPDF.
        Common keys: title, author, subject, creator, producer,
                     creationDate, modDate.
    """
    doc = fitz.open(file_path)
    metadata = doc.metadata  # PyMuPDF returns this as a plain dict
    doc.close()
    return metadata

