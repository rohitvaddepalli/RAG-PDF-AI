import os
import sys
from typing import Any, Dict, Optional
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from components.pdf_uploader import render_uploader, render_pdf_info
from components.qa_view import render_qa_section
from utils.chunker import chunk_extracted_pdf
from utils.embeddings import generate_embeddings
from utils.vector_store import VectorStore


def index_pdf_document(pdf_data: Dict[str, Any]) -> VectorStore:
    """Extract chunks, compute dense embeddings, and populate a VectorStore."""
    if not pdf_data:
        return VectorStore()

    chunks = chunk_extracted_pdf(pdf_data)
    if not chunks:
        return VectorStore()

    embedded_chunks = generate_embeddings(chunks)
    store = VectorStore()
    store.add_chunks(embedded_chunks)
    return store


def main():
    st.set_page_config(
        page_title="Personal PDF AI Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar: Configurations and Controls
    with st.sidebar:
        st.title("⚙️ Assistant Settings")
        st.markdown("---")

        # API Key Configuration
        env_key = os.environ.get("GEMINI_API_KEY", "")
        api_key_input = st.text_input(
            label="Gemini API Key:",
            value=env_key,
            type="password",
            placeholder="AIza...",
            help="Your key is used locally and never stored externally. Auto-loaded from .env if present.",
        )
        effective_key = api_key_input.strip() if api_key_input else None

        if effective_key:
            st.success("🔑 Key configured", icon="✅")
        else:
            st.warning("⚠️ No Gemini API key detected. Set GEMINI_API_KEY in .env or enter above.", icon="⚠️")

        st.markdown("---")
        # Model Selection
        model_options = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-3.5-flash-lite",
        ]
        default_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        model_idx = model_options.index(default_model) if default_model in model_options else 0
        selected_model = st.selectbox(
            "Gemini Model:",
            options=model_options,
            index=model_idx,
            help="Select which Gemini model will formulate fact-grounded responses.",
        )

        st.markdown("---")
        st.markdown("**Retrieval Settings:**")
        top_k = st.slider(
            "Top Chunks (Top-K):",
            min_value=1,
            max_value=10,
            value=4,
            step=1,
            help="Number of context chunks to retrieve.",
        )
        score_threshold = st.slider(
            "Similarity Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum cosine similarity required.",
        )

        st.markdown("---")
        if st.button("🔄 Reset Document & Chat", use_container_width=True):
            st.session_state["uploader_key"] = st.session_state.get("uploader_key", 0) + 1
            for k in [
                "pdf_data",
                "cached_pdf_data",
                "cached_upload_key",
                "vector_store",
                "indexed_file",
                "query_history",
                "active_query_result",
                "user_query_input",
                "suggested_prompt",
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

        st.markdown("---")
        st.caption("Built with Streamlit, PyMuPDF, SentenceTransformers & Google GenAI")

    # Main App Header
    st.title("Personal PDF AI Assistant 🤖")
    st.write("Upload any PDF document and ask questions with verified, instant source citations.")
    st.markdown("---")

    # PDF Uploader Section
    pdf_data = render_uploader()

    if pdf_data is not None:
        st.markdown("---")
        render_pdf_info(pdf_data)
        st.session_state["pdf_data"] = pdf_data

        # Indexing step: cache vector store per file
        current_file_path = pdf_data.get("file_path")
        indexed_path = st.session_state.get("indexed_file")

        if indexed_path != current_file_path or "vector_store" not in st.session_state:
            with st.spinner("Processing document chunks & generating semantic embeddings..."):
                store = index_pdf_document(pdf_data)
                st.session_state["vector_store"] = store
                st.session_state["indexed_file"] = current_file_path
                st.session_state["query_history"] = []
                st.session_state["active_query_result"] = None

        st.markdown("---")
        has_text = pdf_data.get("has_extractable_text", False)
        if not has_text:
            st.error("⚠️ This document has no extractable text. Questions cannot be answered from an empty or scanned image document.")
        else:
            render_qa_section(
                vector_store=st.session_state["vector_store"],
                api_key=effective_key,
                model_name=selected_model,
                top_k=top_k,
                score_threshold=score_threshold,
            )


if __name__ == "__main__":
    main()
