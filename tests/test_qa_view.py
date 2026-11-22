import pytest
from components.qa_view import format_citation, record_query_history

def test_format_citation_fields():
    chunk = {
        "chunk_id": "doc_p1_c0",
        "page_number": 2,
        "similarity": 0.8423,
        "text": "This is sample extracted text from page 2.",
        "source": "sample.pdf",
    }
    citation = format_citation(chunk, rank=1)
    assert citation["rank"] == 1
    assert citation["page_number"] == 2
    assert citation["similarity_pct"] == 84
    assert citation["text"] == "This is sample extracted text from page 2."
    assert citation["source"] == "sample.pdf"

def test_format_citation_handles_edge_cases():
    # Null or missing similarity/page
    chunk = {
        "text": "Chunk with default attributes"
    }
    citation = format_citation(chunk, rank=3)
    assert citation["rank"] == 3
    assert citation["page_number"] == "Unknown"
    assert citation["similarity_pct"] == 0
    assert citation["source"] == "Document"

    # Negative similarity
    chunk_neg = {"similarity": -0.2, "page_number": 1, "text": "Negative similarity test"}
    citation_neg = format_citation(chunk_neg, rank=2)
    assert citation_neg["similarity_pct"] == 0

    # Similarity clamped at 1.0 (100%)
    chunk_high = {"similarity": 1.25, "page_number": 5, "text": "High similarity test"}
    citation_high = format_citation(chunk_high, rank=1)
    assert citation_high["similarity_pct"] == 100

def test_record_query_history_prepends_latest():
    history = []
    res1 = {"answer": "Answer 1", "retrieved_chunks": [], "status": "success"}
    history = record_query_history(history, "Question 1", res1)
    assert len(history) == 1
    assert history[0]["question"] == "Question 1"
    assert history[0]["result"]["answer"] == "Answer 1"

    res2 = {"answer": "Answer 2", "retrieved_chunks": [], "status": "success"}
    history = record_query_history(history, "Question 2", res2)
    assert len(history) == 2
    assert history[0]["question"] == "Question 2"
    assert history[1]["question"] == "Question 1"

def test_record_query_history_limits_max_entries():
    history = []
    for i in range(25):
        history = record_query_history(history, f"Q{i}", {"answer": f"A{i}"}, max_history=10)
    assert len(history) == 10
    assert history[0]["question"] == "Q24"
