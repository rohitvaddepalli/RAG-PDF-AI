"""Integration tests for FastAPI endpoints."""

from unittest.mock import patch
import fitz
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.embedding_service import embedding_service


@pytest.fixture(scope="module", autouse=True)
def init_test_embedding_model():
    embedding_service.load_model()


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def create_dummy_pdf(content: str = "FastAPI powers high performance REST APIs.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), content)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["embedding_model_loaded"] is True
    assert "documents_in_memory" in data


def test_upload_invalid_file_type(client):
    files = {"file": ("test.txt", b"plain text", "text/plain")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_upload_empty_file(client):
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_upload_valid_pdf(client):
    pdf_bytes = create_dummy_pdf("FastAPI and SentenceTransformers work seamlessly together.")
    files = {"file": ("document.pdf", pdf_bytes, "application/pdf")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert "document_id" in data
    assert data["page_count"] == 1
    assert data["chunk_count"] >= 1
    assert data["file_name"] == "document.pdf"


def test_ask_missing_document(client):
    response = client.post("/api/ask", json={
        "document_id": "non_existent_doc",
        "question": "What is FastAPI?",
    })
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_ask_empty_question(client):
    pdf_bytes = create_dummy_pdf("Machine learning models.")
    upload_res = client.post("/api/upload", files={"file": ("ml.pdf", pdf_bytes, "application/pdf")})
    doc_id = upload_res.json()["document_id"]

    response = client.post("/api/ask", json={
        "document_id": doc_id,
        "question": "   ",
    })
    assert response.status_code == 400


def test_ask_below_threshold_returns_fallback_without_llm(client):
    pdf_bytes = create_dummy_pdf("Ingredients: Flour, sugar, butter, and vanilla extract for baking cakes.")
    upload_res = client.post("/api/upload", files={"file": ("recipe.pdf", pdf_bytes, "application/pdf")})
    doc_id = upload_res.json()["document_id"]

    with patch("app.services.llm_service.llm_service.generate_answer") as mock_llm:
        response = client.post("/api/ask", json={
            "document_id": doc_id,
            "question": "What are quantum gravity tensor fluctuations in astrophysical black holes?",
            "top_k": 4,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "I couldn't find this in the document"
        assert data["sources"] == []
        mock_llm.assert_not_called()


def test_ask_with_match_calls_llm(client):
    pdf_bytes = create_dummy_pdf("Antigravity AI is an autonomous agentic system created by Google.")
    upload_res = client.post("/api/upload", files={"file": ("agent.pdf", pdf_bytes, "application/pdf")})
    doc_id = upload_res.json()["document_id"]

    with patch("app.services.llm_service.llm_service.generate_answer", return_value="Antigravity is an AI agent system by Google.") as mock_llm:
        response = client.post("/api/ask", json={
            "document_id": doc_id,
            "question": "What is Antigravity AI?",
            "top_k": 4,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Antigravity is an AI agent system by Google."
        assert len(data["sources"]) > 0
        assert data["sources"][0]["page"] == 1
        mock_llm.assert_called_once()


def test_ask_overview_question_calls_llm(client):
    pdf_bytes = create_dummy_pdf("Quarterly engineering report covering latency, uptime, and deployment goals.")
    upload_res = client.post("/api/upload", files={"file": ("report.pdf", pdf_bytes, "application/pdf")})
    doc_id = upload_res.json()["document_id"]

    for question in [
        "What is this pdf about",
        "Summarize the key takeaways and main points.",
    ]:
        with patch("app.services.llm_service.llm_service.generate_answer", return_value="Engineering report summary.") as mock_llm:
            response = client.post("/api/ask", json={
                "document_id": doc_id,
                "question": question,
                "top_k": 4,
            })
            assert response.status_code == 200
            data = response.json()
            assert data["answer"] == "Engineering report summary."
            assert len(data["sources"]) > 0
            mock_llm.assert_called_once()
