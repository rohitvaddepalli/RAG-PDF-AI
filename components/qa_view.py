from datetime import datetime
import html
from typing import Any, Dict, List, Optional
import streamlit as st

from utils.qa_engine import answer_question, INSUFFICIENT_CONTEXT_MESSAGE
from utils.vector_store import VectorStore


def format_citation(chunk: Dict[str, Any], rank: int = 1) -> Dict[str, Any]:
    """Format a raw retrieved chunk into structured citation metadata."""
    page_number = chunk.get("page_number", "Unknown")
    raw_sim = chunk.get("similarity", 0.0)
    
    try:
        sim_float = float(raw_sim)
    except (ValueError, TypeError):
        sim_float = 0.0

    # Clamp similarity percentage between 0 and 100
    clamped_sim = max(0.0, min(1.0, sim_float))
    similarity_pct = int(round(clamped_sim * 100))

    return {
        "rank": rank,
        "page_number": page_number,
        "similarity_pct": similarity_pct,
        "raw_similarity": sim_float,
        "source": chunk.get("source", "Document"),
        "chunk_id": chunk.get("chunk_id", ""),
        "text": chunk.get("text", "").strip(),
    }


def record_query_history(
    history: List[Dict[str, Any]],
    query: str,
    result: Dict[str, Any],
    max_history: int = 20,
) -> List[Dict[str, Any]]:
    """Prepend the latest question-result pair to session history."""
    new_entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "question": query,
        "result": result,
    }
    updated_history = [new_entry] + (history or [])
    return updated_history[:max_history]


def render_citation_card(citation: Dict[str, Any]) -> None:
    """Render a single source citation card with metrics and excerpt."""
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown(f"**Rank:** `#{citation['rank']}`")
    with col2:
        st.markdown(f"**Location:** `Page {citation['page_number']}`")
    with col3:
        st.markdown(f"**Match:** `{citation['similarity_pct']}% relevance`")

    # Display text snippet in a styled block
    st.markdown(
        f"""
        <div style="background-color: rgba(128, 128, 128, 0.08); padding: 12px; border-radius: 6px; border-left: 4px solid #4CAF50; margin-bottom: 12px;">
            <p style="margin: 0; font-size: 0.92rem; font-style: italic;">
                "{html.escape(citation['text'])}"
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_qa_section(
    vector_store: VectorStore,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
    top_k: int = 4,
    score_threshold: float = 0.25,
) -> None:
    """Render the single-query Q&A studio with instant source citations and session history."""
    st.header("💬 Ask a Question")
    st.caption("Ask specific questions about the uploaded PDF. Answers are strictly grounded in retrieved document excerpts.")

    # Upfront API key check
    if not api_key:
        st.warning("⚠️ No Gemini API key detected. Please configure GEMINI_API_KEY in .env or enter it in the sidebar before asking questions.")

    # Initialize session state keys if absent
    if "query_history" not in st.session_state:
        st.session_state["query_history"] = []
    if "active_query_result" not in st.session_state:
        st.session_state["active_query_result"] = None
    if "user_query_input" not in st.session_state:
        st.session_state["user_query_input"] = ""

    # Quick Prompt Pills
    st.markdown("**Quick Prompts:**")
    pill_col1, pill_col2, pill_col3 = st.columns(3)
    with pill_col1:
        if st.button("📌 Summarize Key Takeaways", use_container_width=True):
            st.session_state["user_query_input"] = "Summarize the key takeaways and main points of this document."
            st.rerun()
    with pill_col2:
        if st.button("📊 Extract Core Findings", use_container_width=True):
            st.session_state["user_query_input"] = "What are the most significant findings, metrics, or conclusions?"
            st.rerun()
    with pill_col3:
        if st.button("🎯 Next Steps & Actions", use_container_width=True):
            st.session_state["user_query_input"] = "What next steps, recommendations, or actions are outlined in this document?"
            st.rerun()

    # Query Input Form
    with st.form("single_query_form", clear_on_submit=False):
        user_query = st.text_area(
            label="Enter your question:",
            key="user_query_input",
            placeholder="e.g., What are the main objectives described in Section 2?",
            height=90,
        )
        submitted = st.form_submit_button("🔎 Ask AI Assistant", use_container_width=True)

    if submitted:
        clean_query = user_query.strip() if user_query else ""
        if not clean_query:
            st.warning("⚠️ Please enter a question to ask the document.")
        else:
            with st.spinner("Analyzing document and generating answer…"):
                res = answer_question(
                    question=clean_query,
                    vector_store=vector_store,
                    api_key=api_key,
                    model_name=model_name,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                st.session_state["active_query_result"] = {
                    "question": clean_query,
                    "result": res,
                }
                st.session_state["query_history"] = record_query_history(
                    history=st.session_state["query_history"],
                    query=clean_query,
                    result=res,
                )

    # Active Query Result & Instant Citations
    active = st.session_state.get("active_query_result")
    if active:
        q_text = active["question"]
        res = active["result"]
        status = res.get("status", "unknown")
        answer_text = res.get("answer", "")
        retrieved_chunks = res.get("retrieved_chunks", [])

        st.markdown("---")
        st.subheader("💡 Answer")
        st.markdown(f"**Question:** *{q_text}*")

        if status == "error":
            st.error(f"❌ {answer_text}")
            if res.get("error_message"):
                st.caption(f"Details: {res['error_message']}")
        elif status == "insufficient_context":
            st.warning(f"ℹ️ {answer_text}")
        else:
            st.success("✅ Answer generated from verified context:")
            st.markdown(
                f"""
                <div style="background-color: rgba(76, 175, 80, 0.08); padding: 16px; border-radius: 8px; border: 1px solid rgba(76, 175, 80, 0.25); margin-bottom: 16px; font-size: 1.05rem; line-height: 1.6;">
                    {html.escape(answer_text).replace('\n', '<br/>')}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Instant Source Citations
        st.markdown("### 📑 Instant Source Citations")
        if not retrieved_chunks:
            st.info("No matching context chunks met the similarity threshold for this query.")
        else:
            st.caption(f"Retrieved {len(retrieved_chunks)} supporting chunk(s) from the document:")
            for idx, raw_chunk in enumerate(retrieved_chunks, start=1):
                citation = format_citation(raw_chunk, rank=idx)
                render_citation_card(citation)

    # Collapsible Past Queries Drawer
    history = st.session_state.get("query_history", [])
    if history:
        st.markdown("---")
        with st.expander(f"🕒 Past Queries in this Session ({len(history)})", expanded=False):
            for item in history:
                st.markdown(f"**[{item['timestamp']}] Q:** {item['question']}")
                ans = item['result'].get('answer', '')
                st.markdown(f"> {ans}")
                hist_chunks = item['result'].get('retrieved_chunks', [])
                if hist_chunks:
                    st.caption(f"Referenced {len(hist_chunks)} chunk(s) [Pages: {', '.join(str(c.get('page_number', '?')) for c in hist_chunks)}]")
                st.markdown("---")
