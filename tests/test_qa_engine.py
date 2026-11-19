"""
tests/test_qa_engine.py
-----------------------
Unit tests for utils/qa_engine.py using mock LLM client and synthetic embeddings.

All tests run completely offline and in-memory without downloading neural network
weights or making network calls to Google Gemini API.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from utils.vector_store import VectorStore
from utils.embeddings import EMBEDDING_DIMENSION
from utils.qa_engine import (
    DEFAULT_GEMINI_MODEL,
    INSUFFICIENT_CONTEXT_MESSAGE,
    QAEngine,
    answer_question,
    build_prompt,
    get_gemini_client,
)
from tests.test_embeddings import MockSentenceTransformer


def make_vector(val: float, dim: int = EMBEDDING_DIMENSION) -> list[float]:
    """Creates a deterministic synthetic vector."""
    vec = [0.0] * dim
    vec[0] = val
    return vec


class TestQAEngine(unittest.TestCase):

    def setUp(self):
        # Set up an in-memory vector store
        self.store = VectorStore(db_path=":memory:", expected_dim=EMBEDDING_DIMENSION)
        self.embed_model = MockSentenceTransformer()

        # Populate sample chunks using embed_model vectors so similarity is 1.0
        synthetic_vec = [float(x) for x in self.embed_model.encode("sample")]
        self.sample_chunks = [
            {
                "chunk_id": "guide.pdf_p1_c0",
                "text": "The Personal PDF AI Assistant is built using Streamlit and PyMuPDF.",
                "source": "guide.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "word_count": 10,
                "char_count": 67,
                "embedding": synthetic_vec,
            },
            {
                "chunk_id": "guide.pdf_p1_c1",
                "text": "Embeddings are generated with Sentence Transformers all-MiniLM-L6-v2.",
                "source": "guide.pdf",
                "page_number": 1,
                "chunk_index": 1,
                "word_count": 8,
                "char_count": 68,
                "embedding": synthetic_vec,
            },
        ]
        self.store.add_chunks(self.sample_chunks)

        # Mock Gemini client
        self.mock_client = MagicMock()
        self.mock_response = MagicMock()
        self.mock_response.text = "The application is built with Streamlit and PyMuPDF."
        self.mock_client.models.generate_content.return_value = self.mock_response

    def tearDown(self):
        self.store.close()

    def test_successful_question_answering(self):
        """Verify successful retrieval and response generation with injected mock client."""
        result = answer_question(
            question="What is this application built with?",
            vector_store=self.store,
            client=self.mock_client,
            embed_model=self.embed_model,
            top_k=3,
            score_threshold=0.1,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "The application is built with Streamlit and PyMuPDF.")
        self.assertTrue(result["context_used"])
        self.assertEqual(len(result["retrieved_chunks"]), 2)
        self.assertIsNone(result["error_message"])

        # Verify mock client call arguments
        self.mock_client.models.generate_content.assert_called_once()
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], DEFAULT_GEMINI_MODEL)
        self.assertIn("Streamlit and PyMuPDF", call_kwargs["contents"])
        self.assertEqual(call_kwargs["config"].temperature, 0.0)

    def test_insufficient_context_when_store_empty(self):
        """Verify early exit with insufficient-context message if vector store is empty."""
        empty_store = VectorStore(db_path=":memory:")
        try:
            result = answer_question(
                question="Any question?",
                vector_store=empty_store,
                client=self.mock_client,
                embed_model=self.embed_model,
            )

            self.assertEqual(result["status"], "insufficient_context")
            self.assertEqual(result["answer"], INSUFFICIENT_CONTEXT_MESSAGE)
            self.assertFalse(result["context_used"])
            self.assertEqual(result["retrieved_chunks"], [])

            # LLM should not be invoked if context is empty
            self.mock_client.models.generate_content.assert_not_called()
        finally:
            empty_store.close()

    def test_insufficient_context_when_below_score_threshold(self):
        """Verify insufficient-context response when chunks do not meet similarity threshold."""
        result = answer_question(
            question="Unrelated question",
            vector_store=self.store,
            client=self.mock_client,
            embed_model=self.embed_model,
            score_threshold=1.1,  # higher than maximum possible similarity
        )

        self.assertEqual(result["status"], "insufficient_context")
        self.assertEqual(result["answer"], INSUFFICIENT_CONTEXT_MESSAGE)
        self.mock_client.models.generate_content.assert_not_called()

    def test_empty_or_whitespace_question(self):
        """Verify empty question returns error without calling vector store or LLM."""
        res_empty = answer_question("", vector_store=self.store, client=self.mock_client)
        self.assertEqual(res_empty["status"], "error")
        self.assertIn("non-empty", res_empty["error_message"])

        res_space = answer_question("   ", vector_store=self.store, client=self.mock_client)
        self.assertEqual(res_space["status"], "error")

        self.mock_client.models.generate_content.assert_not_called()

    def test_missing_api_key_handling(self):
        """Verify clear error status if no client is injected and environment has no key."""
        with patch.dict(os.environ, {}, clear=True):
            result = answer_question(
                question="What is this built with?",
                vector_store=self.store,
                client=None,  # Do not inject client
                embed_model=self.embed_model,
                score_threshold=0.1,
            )

            self.assertEqual(result["status"], "error")
            self.assertIn("API key", result["error_message"])

    def test_gemini_api_error_handling(self):
        """Verify API errors from Google Gemini are caught cleanly without crashing."""
        failing_client = MagicMock()
        failing_client.models.generate_content.side_effect = RuntimeError("Service temporarily unavailable")

        result = answer_question(
            question="What is this built with?",
            vector_store=self.store,
            client=failing_client,
            embed_model=self.embed_model,
            score_threshold=0.1,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("Service temporarily unavailable", result["error_message"])
        self.assertIn("error occurred", result["answer"].lower())

    def test_build_prompt_formatting(self):
        """Verify prompt properly formats sources, page numbers, and question."""
        chunks = [
            {"source": "report.pdf", "page_number": 3, "text": "Net profits grew by 15%."},
            {"source": "report.pdf", "page_number": 4, "text": "Operating margin was 22%."},
        ]
        prompt = build_prompt("What was the profit growth?", chunks)

        self.assertIn("[Block 1 | Source: report.pdf | Page: 3]", prompt)
        self.assertIn("Net profits grew by 15%.", prompt)
        self.assertIn("[Block 2 | Source: report.pdf | Page: 4]", prompt)
        self.assertIn("Operating margin was 22%.", prompt)
        self.assertIn("User Question: What was the profit growth?", prompt)

    def test_qa_engine_class_interface(self):
        """Verify QAEngine class wraps answer_question cleanly."""
        engine = QAEngine(
            client=self.mock_client,
            model_name=DEFAULT_GEMINI_MODEL,
            top_k=3,
            score_threshold=0.1,
        )
        result = engine.ask(
            question="What is this application built with?",
            vector_store=self.store,
            embed_model=self.embed_model,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["answer"], "The application is built with Streamlit and PyMuPDF.")

    def test_get_gemini_client_uses_env_var(self):
        """Verify get_gemini_client picks up GEMINI_API_KEY from environment."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-secret-key-123"}):
            with patch("google.genai.Client") as mock_genai_class:
                get_gemini_client()
                mock_genai_class.assert_called_once_with(api_key="test-secret-key-123")


if __name__ == "__main__":
    unittest.main()
