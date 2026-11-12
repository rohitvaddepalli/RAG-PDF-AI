import streamlit as st
from utils.pdf_handler import save_uploaded_file, extract_text_from_pdf, get_pdf_metadata


def render_uploader() -> dict | None:
    st.header("📄 Upload your PDF")
    uploaded_file = st.file_uploader(
        label="Drag and drop a PDF, or click Browse",
        type=["pdf"],
        help="Maximum file size: 50 MB",
        max_upload_size=50
    )

    if uploaded_file is None:
        st.info("👆 Upload a PDF to get started.")
        return None

    with st.spinner("Reading your PDF… please wait"):
        file_path = save_uploaded_file(uploaded_file)
        extraction = extract_text_from_pdf(file_path)

        if extraction.get("status") == "error":
            st.error(f"❌ Error reading PDF: {extraction['error_message']}")
            return None

        metadata = get_pdf_metadata(file_path)

    result = {
        "file_path":            file_path,
        "file_name":            uploaded_file.name,
        "file_size":            uploaded_file.size,         
        "page_count":           extraction["page_count"],
        "total_words":          extraction["total_words"],
        "has_extractable_text": extraction["has_extractable_text"],
        "pages":                extraction["pages"],
        "full_text":            extraction["full_text"],
        "metadata":             metadata,
    }

    return result


def render_pdf_info(pdf_data: dict) -> None:
    st.success("✅ PDF loaded successfully!")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📄 Pages", pdf_data["page_count"])

    with col2:
        size_kb = round(pdf_data["file_size"] / 1024, 1)
        st.metric("💾 Size", f"{size_kb} KB")

    with col3:
        word_count = pdf_data.get("total_words", len(pdf_data["full_text"].split()))
        st.metric("📝 Words", f"{word_count:,}")

    if not pdf_data.get("has_extractable_text", True):
        st.warning("⚠️ No selectable text found. This document may be scanned, image-based, or blank.")

    with st.expander("📋 PDF Metadata"):
        meta = pdf_data["metadata"]
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
            if value:
                st.write(f"**{label}:** {value}")
                any_shown = True

        if not any_shown:
            st.write("No metadata found in this PDF.")

    with st.expander("🔍 Extracted Text Preview"):
        if not pdf_data.get("has_extractable_text", bool(pdf_data["full_text"].strip())):
            st.warning("⚠️ No selectable text found. This document may be scanned, image-based, or blank.")
        else:
            pages = pdf_data.get("pages", [])
            if pages and isinstance(pages[0], dict):
                page_labels = [f"Page {p['page_number']} ({p['word_count']} words)" for p in pages]
                selected_idx = st.selectbox(
                    "Select page to preview:",
                    options=range(len(pages)),
                    format_func=lambda i: page_labels[i],
                )
                selected_page = pages[selected_idx]
                st.caption(f"Page {selected_page['page_number']} of {pdf_data['page_count']} • {selected_page['word_count']} words • {selected_page['char_count']} characters")
                st.text_area(
                    label=f"Text content for Page {selected_page['page_number']}",
                    value=selected_page["text"] if selected_page["text"] else "(Empty page)",
                    height=250,
                    disabled=True,
                )
            else:
                preview = pdf_data["full_text"][:1000]
                st.text_area(
                    label="Extracted text",
                    value=preview,
                    height=250,
                    disabled=True,
                    label_visibility="collapsed",
                )