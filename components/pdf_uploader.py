"""
components/pdf_uploader.py
--------------------------
This module owns the entire "upload a PDF" section of the UI.
By separating UI pieces into components/, we can easily reuse or
swap them without touching the main app file.
"""

import streamlit as st
from utils.pdf_handler import save_uploaded_file, extract_text_from_pdf, get_pdf_metadata


def render_uploader() -> dict | None:
    """
    Draw the PDF upload widget and, once a file is uploaded,
    process it and return a structured result dictionary.

    Returns
    -------
    dict or None
        None  → no file uploaded yet.
        dict  → {
                    "file_path"  : str,
                    "file_name"  : str,
                    "file_size"  : int (bytes),
                    "page_count" : int,
                    "pages"      : list[str],
                    "full_text"  : str,
                    "metadata"   : dict,
                 }
    """
    st.header("📄 Upload your PDF")

    # ------------------------------------------------------------------ #
    # st.file_uploader renders a drag-and-drop area.                      #
    # `type=["pdf"]` restricts selection to PDF files only.               #
    # ------------------------------------------------------------------ #
    uploaded_file = st.file_uploader(
        label="Drag and drop a PDF, or click Browse",
        type=["pdf"],
        help="Maximum file size: 50 MB  •  Only PDF files are accepted",
    )

    # Nothing uploaded yet — tell the user what to do and exit early
    if uploaded_file is None:
        st.info("👆 Upload a PDF to get started.")
        return None

    # ------------------------------------------------------------------ #
    # A file was uploaded — show a spinner while we process it.           #
    # st.spinner() is a context manager; code inside runs while the       #
    # spinner is visible on screen.                                        #
    # ------------------------------------------------------------------ #
    with st.spinner("Reading your PDF… please wait"):
        # 1. Save the in-memory file to the uploads/ folder
        file_path = save_uploaded_file(uploaded_file)

        # 2. Extract text from every page
        extraction = extract_text_from_pdf(file_path)

        # 3. Read embedded metadata
        metadata = get_pdf_metadata(file_path)

    # Build the result we'll pass back to app.py
    result = {
        "file_path":  file_path,
        "file_name":  uploaded_file.name,
        "file_size":  uploaded_file.size,          # size in bytes
        "page_count": extraction["page_count"],
        "pages":      extraction["pages"],
        "full_text":  extraction["full_text"],
        "metadata":   metadata,
    }

    return result


def render_pdf_info(pdf_data: dict) -> None:
    """
    Display a neat summary card after a PDF is uploaded successfully.

    Parameters
    ----------
    pdf_data : dict
        The dictionary returned by render_uploader().
    """
    st.success("✅ PDF loaded successfully!")

    # st.columns() splits the page into side-by-side columns.
    # [1, 1, 1] means three equal-width columns.
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 Pages", pdf_data["page_count"])

    with col2:
        # Convert bytes → kilobytes, rounded to 1 decimal place
        size_kb = round(pdf_data["file_size"] / 1024, 1)
        st.metric("💾 Size", f"{size_kb} KB")

    with col3:
        # Count words as a quick readability hint
        word_count = len(pdf_data["full_text"].split())
        st.metric("📝 Words", f"{word_count:,}")  # comma-separated thousands

    # ------------------------------------------------------------------ #
    # Expandable metadata section — hidden by default so the UI stays     #
    # clean, but accessible for curious users.                            #
    # ------------------------------------------------------------------ #
    with st.expander("📋 PDF Metadata"):
        meta = pdf_data["metadata"]

        # Only show fields that actually have a value
        display_fields = {
            "Title":    meta.get("title"),
            "Author":   meta.get("author"),
            "Subject":  meta.get("subject"),
            "Creator":  meta.get("creator"),
            "Producer": meta.get("producer"),
            "Created":  meta.get("creationDate"),
        }

        any_shown = False
        for label, value in display_fields.items():
            if value:  # skip empty / None fields
                st.write(f"**{label}:** {value}")
                any_shown = True

        if not any_shown:
            st.write("No metadata found in this PDF.")

    # ------------------------------------------------------------------ #
    # Text preview — shows the first 1 000 characters of extracted text.  #
    # st.text_area() with disabled=True makes it read-only.               #
    # ------------------------------------------------------------------ #
    with st.expander("🔍 Extracted Text Preview (first 1 000 characters)"):
        preview = pdf_data["full_text"][:1000]
        if preview.strip():
            st.text_area(
                label="Extracted text",
                value=preview,
                height=250,
                disabled=True,       # read-only
                label_visibility="collapsed",
            )
        else:
            st.warning(
                "⚠️ No text could be extracted. "
                "This PDF might be image-based (scanned). "
                "OCR support will be added in a future session."
            )

