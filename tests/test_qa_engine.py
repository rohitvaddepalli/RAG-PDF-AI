import os
import unittest
from unittest.mock import MagicMock, patch

from utils.vector_store import VectorStore
from utils.embeddings import EMBEDDING_DIMENSION
from utils.qa_engine import (
    DEFAULT_GEMINI_MODEL,
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYSTEM_INSTRUCTION,
    QAEngine,
    answer_question,
    build_prompt,
    get_gemini_client,
    sanitize_error_message,
)
from tests.test_embeddings import MockSentenceTransformer


class TestQAEngine(unittest.TestCase):

    def setUp(self):
        self.store = VectorStore(db_path=":memory:", expected_dim=EMBEDDING_DIMENSION)
        self.embed_model = MockSentenceTransformer()

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

        self.mock_client = MagicMock()
        self.mock_response = MagicMock()
        self.mock_response.text = "The application is built with Streamlit and PyMuPDF."
        self.mock_client.models.generate_content.return_value = self.mock_response

    def tearDown(self):
        self.store.close()

    def test_successful_generation(self):
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

        self.mock_client.models.generate_content.assert_called_once()
        call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
        self.assertEqual(call_kwargs["model"], DEFAULT_GEMINI_MODEL)
        self.assertIn("Streamlit and PyMuPDF", call_kwargs["contents"])
        self.assertEqual(call_kwargs["config"].temperature, 0.0)

    def test_prompt_context_structure_and_untrusted_markers(self):
        chunks = [
            {"source": "report.pdf", "page_number": 3, "text": "Net profits grew by 15%."},
            {"source": "report.pdf", "page_number": 4, "text": "Operating margin was 22%."},
        ]
        prompt = build_prompt("What was the profit growth?", chunks)

        self.assertIn("Context Block 1 [Source: report.pdf | Page: 3]", prompt)
        self.assertIn("Net profits grew by 15%.", prompt)
        self.assertIn("Context Block 2 [Source: report.pdf | Page: 4]", prompt)
        self.assertIn("Operating margin was 22%.", prompt)
        self.assertIn("<context>", prompt)
        self.assertIn("</context>", prompt)
        self.assertIn("untrusted", prompt.lower())
        self.assertIn("User Question: What was the profit growth?", prompt)

    def test_system_instruction_treats_context_as_untrusted(self):
        self.assertIn("untrusted", SYSTEM_INSTRUCTION.lower())
        self.assertIn("never as instructions", SYSTEM_INSTRUCTION.lower())
        self.assertIn(INSUFFICIENT_CONTEXT_MESSAGE, SYSTEM_INSTRUCTION)

    def test_empty_and_invalid_questions(self):
        for invalid_q in ["", "   ", "\n\t", None, 12345]:
            result = answer_question(
                question=invalid_q,
                vector_store=self.store,
                client=self.mock_client,
            )
            self.assertEqual(result["status"], "error")
            self.assertIn("non-empty", result["error_message"].lower())
            self.assertFalse(result["context_used"])
            self.assertEqual(result["retrieved_chunks"], [])

        self.mock_client.models.generate_content.assert_not_called()

    def test_no_retrieved_chunks_empty_store(self):
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
            self.mock_client.models.generate_content.assert_not_called()
        finally:
            empty_store.close()

    def test_no_retrieved_chunks_below_score_threshold(self):
        result = answer_question(
            question="Unrelated question",
            vector_store=self.store,
            client=self.mock_client,
            embed_model=self.embed_model,
            score_threshold=1.1,
        )

        self.assertEqual(result["status"], "insufficient_context")
        self.assertEqual(result["answer"], INSUFFICIENT_CONTEXT_MESSAGE)
        self.assertFalse(result["context_used"])
        self.assertEqual(result["retrieved_chunks"], [])
        self.mock_client.models.generate_content.assert_not_called()

    def test_insufficient_evidence_returned_by_model(self):
        self.mock_response.text = INSUFFICIENT_CONTEXT_MESSAGE

        result = answer_question(
            question="What is the CEO salary?",
            vector_store=self.store,
            client=self.mock_client,
            embed_model=self.embed_model,
            score_threshold=0.1,
        )

        self.assertEqual(result["status"], "insufficient_context")
        self.assertEqual(result["answer"], INSUFFICIENT_CONTEXT_MESSAGE)
        self.assertTrue(result["context_used"])

    def test_empty_model_response_fallback(self):
        for empty_text in ["", "   ", None]:
            self.mock_response.text = empty_text
            result = answer_question(
                question="What is this built with?",
                vector_store=self.store,
                client=self.mock_client,
                embed_model=self.embed_model,
                score_threshold=0.1,
            )

            self.assertEqual(result["status"], "insufficient_context")
            self.assertEqual(result["answer"], INSUFFICIENT_CONTEXT_MESSAGE)

    def test_api_quota_error_handling(self):
        quota_client = MagicMock()
        quota_client.models.generate_content.side_effect = RuntimeError(
            "429 Resource has been exhausted (e.g. check quota)."
        )

        result = answer_question(
            question="What is this built with?",
            vector_store=self.store,
            client=quota_client,
            embed_model=self.embed_model,
            score_threshold=0.1,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("quota", result["answer"].lower())
        self.assertIn("429", result["error_message"])

    def test_network_error_handling(self):
        network_client = MagicMock()
        network_client.models.generate_content.side_effect = ConnectionError(
            "Failed to establish a new connection: [Errno 11001] getaddrinfo failed"
        )

        result = answer_question(
            question="What is this built with?",
            vector_store=self.store,
            client=network_client,
            embed_model=self.embed_model,
            score_threshold=0.1,
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("network error", result["answer"].lower())

    def test_missing_configuration_error(self):
        with patch.dict(os.environ, {}, clear=True):
            result = answer_question(
                question="What is this built with?",
                vector_store=self.store,
                client=None,
                embed_model=self.embed_model,
                score_threshold=0.1,
            )

            self.assertEqual(result["status"], "error")
            self.assertIn("GEMINI_API_KEY", result["answer"])
            self.assertIn("API key", result["error_message"])

    def test_secret_safe_error_handling_injected_key(self):
        leaky_client = MagicMock()
        secret_key = "AIzaSyFakeSecretKey123456789012345678"
        leaky_client.models.generate_content.side_effect = RuntimeError(
            f"Call failed for https://generativelanguage.googleapis.com/v1beta/models?key={secret_key}"
        )

        result = answer_question(
            question="What is this built with?",
            vector_store=self.store,
            client=leaky_client,
            api_key=secret_key,
            embed_model=self.embed_model,
            score_threshold=0.1,
        )

        self.assertEqual(result["status"], "error")
        self.assertNotIn(secret_key, result["answer"])
        self.assertNotIn(secret_key, result["error_message"])
        self.assertIn("[REDACTED_API_KEY]", result["error_message"])

    def test_sanitize_error_message_patterns(self):
        raw_msg = (
            "Error on https://example.com/api?key=AIzaSyTestKey1234567890123456789012 "
            "with Authorization: Bearer secret_token_xyz999"
        )
        cleaned = sanitize_error_message(raw_msg)
        self.assertNotIn("AIzaSyTestKey1234567890123456789012", cleaned)
        self.assertNotIn("secret_token_xyz999", cleaned)
        self.assertIn("[REDACTED_API_KEY]", cleaned)
        self.assertIn("[REDACTED_TOKEN]", cleaned)

    def test_model_configuration_from_env(self):
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-1.5-pro"}):
            answer_question(
                question="What is this built with?",
                vector_store=self.store,
                client=self.mock_client,
                embed_model=self.embed_model,
                score_threshold=0.1,
            )

            call_kwargs = self.mock_client.models.generate_content.call_args.kwargs
            self.assertEqual(call_kwargs["model"], "gemini-1.5-pro")

    def test_qa_engine_class_interface(self):
        engine = QAEngine(
            client=self.mock_client,
            model_name="gemini-1.5-flash",
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
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-env-key-xyz"}):
            with patch("google.genai.Client") as mock_genai_class:
                get_gemini_client()
                mock_genai_class.assert_called_once_with(api_key="test-env-key-xyz")

    def test_get_gemini_client_raises_when_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                get_gemini_client()


if __name__ == "__main__":
    unittest.main()
