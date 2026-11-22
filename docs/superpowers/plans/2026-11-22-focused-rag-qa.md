# Focused Single-Query RAG Q&A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the Personal PDF AI Assistant by implementing an end-to-end single-query Q&A interface with instant source citations, session caching, sidebar controls, and project documentation.

**Architecture:** A modular Streamlit presentation layer where `app.py` coordinates sidebar configuration and document indexing, while `components/qa_view.py` renders single-query prompts, sample questions, active grounded answers, instant source citation cards, and a collapsible session query drawer.

**Tech Stack:** Python 3.14 / Streamlit, PyMuPDF, LangChain text-splitters, SentenceTransformers (`all-MiniLM-L6-v2`), Google GenAI SDK (`google-genai`), Pytest.

**Spec:** `docs/superpowers/specs/2026-11-22-focused-rag-qa-design.md`

## Global Constraints

- Must adhere strictly to fact-grounded answering using `utils.qa_engine.answer_question`.
- Never re-embed or re-chunk uploaded documents on every Streamlit rerun; indexing must be guarded by `st.session_state`.
- All exception messages from API calls must be sanitized via `sanitize_error_message`.
- Existing passing tests (53 tests in `tests/`) must remain passing without breakage.
- Test runner must use `.\venv\Scripts\python.exe -m pytest`.

## Review Focus

- Empty or whitespace-only query input: Expect a user-friendly alert without executing embedding or API calls.
- Scanned / zero-text PDF: Expect a warning banner disabling the Q&A section with a clear message.
- Missing or invalid Gemini API key: Expect prominent notice directing user to `.env` or sidebar input, preventing unhandled crashes.
- Unanswerable query not in document: Expect graceful return of `INSUFFICIENT_CONTEXT_MESSAGE` with no source hallucination.
- Switching uploaded PDFs: Expect vector store and session query drawer to automatically reset cleanly to avoid cross-document contamination.

---

### Task 1: Q&A View Component (`components/qa_view.py`) and Unit Tests

**Files:**
- Create: `components/qa_view.py`
- Create: `tests/test_qa_view.py`

**Interfaces:**
- Consumes:
  - `utils.vector_store.VectorStore`
  - `utils.qa_engine.answer_question(question: str, vector_store: VectorStore, ...) -> Dict[str, Any]`
  - `utils.qa_engine.INSUFFICIENT_CONTEXT_MESSAGE`
- Produces:
  - `format_citation(chunk: Dict[str, Any], rank: int) -> Dict[str, Any]`
  - `record_query_history(history: List[Dict[str, Any]], query: str, result: Dict[str, Any]) -> List[Dict[str, Any]]`
  - `render_qa_section(vector_store: VectorStore, api_key: Optional[str] = None, model_name: str = "gemini-2.5-flash", top_k: int = 4, score_threshold: float = 0.25) -> None`

- [ ] **Step 1: Write the failing tests for citation formatting and history tracking**

```python
# tests/test_qa_view.py
from components.qa_view import format_citation, record_query_history

def test_format_citation_fields():
    chunk = {
        "chunk_id": "doc_p1_c0",
        "page_number": 2,
        "similarity": 0.8423,
        "text": "This is sample extracted text from page 2.",
        "source": "sample.pdf"
    }
    citation = format_citation(chunk, rank=1)
    assert citation["rank"] == 1
    assert citation["page_number"] == 2
    assert citation["similarity_pct"] == 84
    assert citation["text"] == "This is sample extracted text from page 2."

def test_record_query_history_prepends_latest():
    history = []
    res1 = {"answer": "Answer 1", "retrieved_chunks": [], "status": "success"}
    history = record_query_history(history, "Question 1", res1)
    assert len(history) == 1
    assert history[0]["question"] == "Question 1"

    res2 = {"answer": "Answer 2", "retrieved_chunks": [], "status": "success"}
    history = record_query_history(history, "Question 2", res2)
    assert len(history) == 2
    assert history[0]["question"] == "Question 2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_qa_view.py -v`  
Expected: FAIL with `ModuleNotFoundError` or `ImportError: cannot import name 'format_citation'`

- [ ] **Step 3: Implement `components/qa_view.py`**

Implement helper functions:
- `format_citation(chunk: Dict[str, Any], rank: int) -> Dict[str, Any]` computing rank, page number, similarity percentage, and snippet text.
- `record_query_history(history: List[Dict[str, Any]], query: str, result: Dict[str, Any]) -> List[Dict[str, Any]]` appending/prepending timestamped query entries.
- `render_qa_section(vector_store: VectorStore, api_key: Optional[str], model_name: str, top_k: int, score_threshold: float)`:
  - Renders query input form with text area or text input.
  - Provides quick-ask sample buttons ("Summarize key takeaways", "What are the key findings?", "Extract conclusions and next steps").
  - On submit: validates query, invokes `answer_question`, stores in `st.session_state["query_history"]`, and updates active result.
  - Renders Answer box with appropriate status badge (`success`, `insufficient_context`, `error`).
  - Renders Instant Citations directly below answer with page badge, score badge, and chunk text.
  - Renders collapsible past queries accordion (`🕒 Past Queries in this Session`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_qa_view.py -v`  
Expected: PASS

- [ ] **Step 5: Commit Task 1**

```bash
git add components/qa_view.py tests/test_qa_view.py
git commit -m "feat: implement single-query QA view component and unit tests"
```

---

### Task 2: Application Pipeline Integration & Caching in `app.py`

**Files:**
- Modify: `app.py`
- Create: `tests/test_app_integration.py`

**Interfaces:**
- Consumes:
  - `components.pdf_uploader.render_uploader`, `render_pdf_info`
  - `components.qa_view.render_qa_section`
  - `utils.chunker.chunk_extracted_pdf`
  - `utils.embeddings.generate_embeddings`
  - `utils.vector_store.VectorStore`
- Produces:
  - `index_pdf_document(pdf_data: Dict[str, Any]) -> VectorStore` (cached indexing helper)

- [ ] **Step 1: Write integration tests for document indexing helper**

```python
# tests/test_app_integration.py
import pytest
from app import index_pdf_document
from utils.vector_store import VectorStore

def test_index_pdf_document_creates_populated_vector_store():
    fake_pdf_data = {
        "file_path": "uploads/fake.pdf",
        "file_name": "fake.pdf",
        "pages": [
            {"page_number": 1, "text": "Artificial Intelligence is transforming search and retrieval applications."},
            {"page_number": 2, "text": "Vector databases enable semantic similarity search across dense embeddings."}
        ]
    }
    store = index_pdf_document(fake_pdf_data)
    assert isinstance(store, VectorStore)
    assert store.count() > 0
    assert store.chunks[0]["page_number"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_app_integration.py -v`  
Expected: FAIL with `ImportError: cannot import name 'index_pdf_document'`

- [ ] **Step 3: Implement `index_pdf_document` and wire `app.py`**

- In `app.py`:
  - Define `index_pdf_document(pdf_data: Dict[str, Any]) -> VectorStore`: chunks the document, embeds the chunks, creates a `VectorStore`, adds chunks, and returns it.
  - Setup sidebar controls:
    - Gemini API Key input (defaults to `os.environ.get("GEMINI_API_KEY", "")`, masked, with status indicator "🔑 Key Configured" or "⚠️ Missing API Key").
    - Model selector dropdown: `gemini-2.5-flash` (default), `gemini-2.5-pro`, `gemini-1.5-flash`, `gemini-1.5-pro`.
    - Retrieval tuning: Top-K slider (default 4, range 1-10), Similarity threshold slider (default 0.25, range 0.0-1.0).
    - Reset button: clears `st.session_state` keys (`vector_store`, `query_history`, `indexed_file`).
  - When `pdf_data` is returned:
    - Check if `st.session_state.get("indexed_file") != pdf_data["file_path"]`:
      - If different or unindexed: run `index_pdf_document` with `st.spinner("Processing document chunks & generating embeddings...")`, save store to `st.session_state["vector_store"]`, set `st.session_state["indexed_file"] = pdf_data["file_path"]`, reset `query_history = []`.
    - Render `render_pdf_info(pdf_data)`.
    - If `pdf_data.get("has_extractable_text")`:
      - Call `render_qa_section(st.session_state["vector_store"], api_key=effective_key, model_name=selected_model, top_k=top_k, score_threshold=score_threshold)`.
    - Else: show warning that no text is extractable for Q&A.

- [ ] **Step 4: Run test to verify it passes**

Run: `.\venv\Scripts\python.exe -m pytest tests/test_app_integration.py -v`  
Expected: PASS

- [ ] **Step 5: Run full test suite across all modules**

Run: `.\venv\Scripts\python.exe -m pytest -v`  
Expected: All 55+ tests PASS

- [ ] **Step 6: Commit Task 2**

```bash
git add app.py tests/test_app_integration.py
git commit -m "feat: wire end-to-end PDF indexing pipeline and sidebar controls in app.py"
```

---

### Task 3: Configuration, Documentation & Final Verification

**Files:**
- Create: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Create `.env.example`**

Add template entries with comments explaining each setting:
```
# Gemini API Key for question answering (obtain from Google AI Studio)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Default Gemini model (e.g. gemini-2.5-flash, gemini-2.5-pro)
GEMINI_MODEL=gemini-2.5-flash
```

- [ ] **Step 2: Update `README.md`**

Write clear documentation containing:
- Project title & description
- Architecture & key technologies diagram
- Features (PyMuPDF extraction, Recursive text chunking, local SentenceTransformer embeddings, in-memory cosine VectorStore, fact-grounded Gemini answering with instant citations, collapsible session history)
- Setup & Installation guide (Python venv, requirements installation)
- Environment configuration (`.env` setup)
- Running the application (`streamlit run app.py`)
- Running the test suite (`python -m pytest`)

- [ ] **Step 3: Run full pytest suite to verify zero regressions**

Run: `.\venv\Scripts\python.exe -m pytest -v`  
Expected: 100% test pass rate

- [ ] **Step 4: Commit Task 3**

```bash
git add .env.example README.md
git commit -m "docs: add .env.example and comprehensive README documentation"
```
