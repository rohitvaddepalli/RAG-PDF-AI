# Design Specification: Focused Single-Query RAG Q&A with Instant Citations

**Date:** 2026-11-22  
**Status:** In Review  
**Topic:** End-to-End Streamlit Application Integration for RAG PDF AI  

---

## 1. Intent & Purpose
The goal is to complete the **Personal PDF AI Assistant** by integrating the fully-implemented and unit-tested backend utilities (`pdf_handler`, `chunker`, `embeddings`, `vector_store`, and `qa_engine`) into an intuitive, polished, and fact-grounded Streamlit interface.

Users need to upload a PDF document and immediately ask specific, factual questions. The application must retrieve semantically relevant context chunks, submit a grounded query to Google Gemini, and instantly display the verified answer alongside clear, transparent source citations (page number, similarity score, exact text excerpt).

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Automated Ingestion & Caching**: When a PDF is uploaded, automatically parse pages, chunk text with `RecursiveCharacterTextSplitter`, compute embeddings via `all-MiniLM-L6-v2`, and index chunks into a `VectorStore`. Cache the indexed store in `st.session_state` using the file's unique hash or path so subsequent questions execute instantly without re-embedding.
- **Focused Single-Query UI**: A clean question submission form with quick-ask suggestions ("Summarize key takeaways", "What are the main requirements/conclusions?", etc.).
- **Instant Source Citations**: Display retrieved source chunks immediately beneath the answer. Each citation card displays:
  - Page badge (e.g., `Page 3`)
  - Similarity relevance percentage (e.g., `86% match`)
  - Exact quoted chunk snippet
- **Collapsible History Drawer**: A lightweight accordion below the active query result preserving past queries and answers from the current session for easy reference.
- **Robust Sidebar Controls**:
  - API Key handling: Auto-detect from `.env` (`GEMINI_API_KEY`), with a fallback sidebar input to enter or override it with visual status indicator.
  - Model selection dropdown: Options for `gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-1.5-flash`, etc.
  - Hyperparameter tuning: Sliders for `Top-K` retrieved chunks (1–10) and minimum similarity score threshold (0.0–1.0).
  - Clear / Reset action: Button to reset document index and ask history.
- **Documentation & Configuration Artifacts**:
  - `.env.example` documenting all configuration keys.
  - Comprehensive `README.md` covering setup, architecture, and testing.

### Non-Goals
- Multi-turn conversational chat that threads context from previous questions into the current LLM prompt. (The user explicitly specified a focused single-query Q&A format).
- Live PDF page visual viewer/canvas (extracted text preview and metadata preview are already supported and sufficient).
- External vector database deployment (local in-memory/JSON `VectorStore` is already fast, self-contained, and tested).

### Anti-Goals
- **Hallucinated or Speculative Answers**: Answers must strictly adhere to the retrieved text or yield the designated `INSUFFICIENT_CONTEXT_MESSAGE`.
- **Redundant Embeddings Computation**: Under no circumstances should the system re-embed the PDF on every Streamlit script re-run. Indexing must be guarded by session state caching.
- **API Key Leakage**: Any exception messages from API calls must pass through `sanitize_error_message`.

---

## 3. Approaches Considered

1. **Approach 1: Dual-Pane Focused Studio (Selected)**
   - Top area provides a clean query form and sample prompt pills.
   - Answer banner prominently presents Gemini's grounded answer.
   - Instant Sources section directly follows the answer with rank, page number, similarity %, and exact chunk text.
   - *Enhancement*: Includes a collapsible past-query drawer (from Approach 3) so previous questions are retained without cluttering the screen.
   - *Rationale*: Delivers the exact workflow requested—focused single-query analysis with zero extra clicks needed to inspect citations.

2. **Approach 2: Tabbed Single-Query View**
   - Segregates the answer into Tab 1 and source chunks into Tab 2.
   - *Tradeoff*: Hides citations behind an extra tab click, hurting immediate verifiability.

3. **Approach 3: Collapsible Feed / Full History Log**
   - Appends all past queries and answers sequentially down the page.
   - *Tradeoff*: Causes excessive scrolling over a long session, shifting focus away from the active query.

---

## 4. System Architecture & Component Design

```
+------------------------------------------------------------------+
|                            app.py                                |
|  - Streamlit Page Configuration & Theming                        |
|  - Sidebar: API Key, Model Picker, Top-K & Threshold Sliders,    |
|             Reset Index                                          |
|  - Upload Section: components/pdf_uploader.py                    |
|  - Ingestion Orchestration: chunk_extracted_pdf -> embeddings -> |
|             VectorStore (cached in st.session_state)             |
|  - Q&A Studio: components/qa_view.py                             |
+------------------------------------------------------------------+
               |                                    |
               v                                    v
   +-----------------------+            +-----------------------+
   |   components/         |            |        utils/         |
   |   - pdf_uploader.py   |            |   - pdf_handler.py    |
   |   - qa_view.py        |            |   - chunker.py        |
   +-----------------------+            |   - embeddings.py     |
                                        |   - vector_store.py   |
                                        |   - qa_engine.py      |
                                        +-----------------------+
```

### Component Details
1. **`components/qa_view.py`**:
   - `render_qa_section(vector_store: VectorStore, api_key: Optional[str], model_name: str, top_k: int, score_threshold: float)`
   - Form with input text, sample question buttons, and "Ask Document" submit button.
   - Execution handler invoking `answer_question` from `utils.qa_engine`.
   - Result card showing Answer with status badges (e.g. `Success`, `Insufficient Context`, `Error`).
   - Citation cards rendering each retrieved chunk:
     - Rank & Page badge
     - Cosine similarity percentage badge
     - Full chunk text in styled block
   - Collapsible "Past Queries in this Session" drawer showing previous Q&A pairs with their citations.

2. **`app.py` Enhancements**:
   - Reads `.env` on load via `load_dotenv()`.
   - Sidebar renders API key input (pre-filled if present in environment, masked).
   - Sidebar renders Model selection (`gemini-2.5-flash`, `gemini-2.5-pro`, `gemini-3.5-flash-lite`, etc.) and Retrieval Tuning knobs.
   - Automatically handles ingestion when `pdf_data` is available:
     - Checks if `st.session_state["vector_store"]` exists for the current `pdf_data["file_path"]`.
     - If not, shows a progress spinner/progress bar:
       - 1. Chunking pages (`chunk_extracted_pdf`)
       - 2. Generating embeddings (`generate_embeddings`)
       - 3. Populating `VectorStore` (`store.add_chunks`)
       - 4. Storing in `st.session_state["vector_store"]`
   - Replaces the placeholder message with `render_qa_section`.

3. **Configuration & Documentation**:
   - `.env.example`: Provides `GEMINI_API_KEY`, `GEMINI_MODEL`.
   - `README.md`: Complete project documentation covering requirements, architecture, setup instructions, and testing guidelines.

---

## 5. Error Handling & Edge Cases
- **No Text in PDF / Scanned PDF**: If `has_extractable_text` is `False`, prompt the user that no selectable text was found and disable the Q&A section.
- **Missing API Key**: Clearly alert the user in both the sidebar and the Q&A panel with instructions to either create a `.env` file or paste the key directly in the sidebar.
- **Insufficient Context**: If the query topic is not discussed in the PDF, gracefully display the system message: *"I do not have enough information from the uploaded document to answer this question."*
- **Empty / Blank Query**: Validate and warn the user without calling the backend.
- **API Errors / Rate Limits (429) / Network issues**: Routed through `classify_api_error` and `sanitize_error_message` ensuring no raw tokens or keys appear on screen.

---

## 6. What's Left to the Builder
- Precise CSS styling for the source citation cards to ensure clean visual hierarchy in Streamlit (using native `st.container`, `st.metric`, `st.expander`, or markdown cards).
- Adding integration/unit tests for `qa_view` logic.
