import streamlit as st
from components.pdf_uploader import render_uploader, render_pdf_info

st.set_page_config(
    page_title="Personal PDF AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("📚 PDF AI Assistant")
    st.markdown("---")
    st.markdown(
        """
        **How it works (today):**
        1. Upload a PDF  
        2. The app reads every page  
        3. Text is extracted and shown  

        *RAG-based Q&A coming in the next session!*
        """
    )
    st.markdown("---")
    st.caption("Built with Streamlit + PyMuPDF")


st.title("Personal PDF AI Assistant 🤖")
st.write("Ask questions about any PDF — powered by AI.")
st.markdown("---")
pdf_data = render_uploader()

if pdf_data is not None:
    st.markdown("---")
    render_pdf_info(pdf_data)
    st.session_state["pdf_data"] = pdf_data

    st.markdown("---")
    st.header("💬 Ask a Question")
    st.info(
        "🚧 The AI Q&A feature is coming in the next session! "
        "For now, explore the extracted text preview above."
    )
