"""
tests/test_embeddings.py
-------------------------
Unit tests for utils/embeddings.py using a mock embedding model.

All tests run completely offline and in-memory without downloading
neural network weights from Hugging Face or requiring network access.
"""

import math
import unittest
from unittest.mock import MagicMock
import numpy as np

from utils.embeddings import (
    DEFAULT_MODEL_NAME,
    EMBEDDING_DIMENSION,
    generate_embeddings,
    embed_query,
    load_embedding_model,
    validate_embeddings,
)


class MockSentenceTransformer:
    """Mock SentenceTransformer that returns deterministic 384-d vectors."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        self.dimension = dimension
        self.encode_calls = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        self.encode_calls.append({
            "texts": texts,
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
        })
        if isinstance(texts, str):
            return np.ones(self.dimension, dtype=np.float32) * 0.42
        return np.ones((len(texts), self.dimension), dtype=np.float32) * 0.42


class TestEmbeddingsModule(unittest.TestCase):

    def setUp(self):
        self.mock_model = MockSentenceTransformer()
        self.sample_chunks = [
            {
                "chunk_id": "doc.pdf_p1_c0",
                "text": "First chunk text from introduction.",
                "page_number": 1,
                "source": "doc.pdf",
                "chunk_index": 0,
                "word_count": 5,
                "char_count": 35,
            },
            {
                "chunk_id": "doc.pdf_p1_c1",
                "text": "Second chunk discussing AI methodology.",
                "page_number": 1,
                "source": "doc.pdf",
                "chunk_index": 1,
                "word_count": 5,
                "char_count": 39,
            },
        ]

    def test_generate_embeddings_success(self):
        """Test standard embedding generation with model injection."""
        results = generate_embeddings(self.sample_chunks, model=self.mock_model)

        self.assertEqual(len(results), 2)

        for original, augmented in zip(self.sample_chunks, results):
            # Verify original metadata is preserved
            self.assertEqual(augmented["chunk_id"], original["chunk_id"])
            self.assertEqual(augmented["text"], original["text"])
            self.assertEqual(augmented["page_number"], original["page_number"])
            self.assertEqual(augmented["source"], original["source"])
            self.assertEqual(augmented["word_count"], original["word_count"])
            self.assertEqual(augmented["char_count"], original["char_count"])

            # Verify embedding structure
            self.assertIn("embedding", augmented)
            self.assertIsInstance(augmented["embedding"], list)
            self.assertEqual(len(augmented["embedding"]), EMBEDDING_DIMENSION)
            self.assertIsInstance(augmented["embedding"][0], float)

    def test_batch_encoding_is_utilized(self):
        """Ensure batch encoding is performed in a single forward pass."""
        custom_batch_size = 16
        generate_embeddings(
            self.sample_chunks,
            model=self.mock_model,
            batch_size=custom_batch_size,
        )

        self.assertEqual(len(self.mock_model.encode_calls), 1)
        call_info = self.mock_model.encode_calls[0]
        self.assertEqual(len(call_info["texts"]), 2)
        self.assertEqual(call_info["batch_size"], custom_batch_size)

    def test_empty_and_none_input(self):
        """Empty or None input lists should return an empty list without error."""
        self.assertEqual(generate_embeddings([], model=self.mock_model), [])
        self.assertEqual(generate_embeddings(None, model=self.mock_model), [])

    def test_invalid_and_missing_text_handling(self):
        """Safely skip chunks that lack valid text while preserving valid ones."""
        mixed_chunks = [
            {"chunk_id": "c1", "text": "Valid text here.", "page_number": 1},
            {"chunk_id": "c2", "text": "", "page_number": 1},            # empty text
            {"chunk_id": "c3", "text": "   ", "page_number": 1},          # whitespace only
            {"chunk_id": "c4", "page_number": 1},                          # missing text key
            {"chunk_id": "c5", "text": None, "page_number": 1},           # None text
            "not a dictionary",                                            # invalid item
            {"chunk_id": "c6", "text": "Another valid chunk.", "page_number": 2},
        ]

        results = generate_embeddings(mixed_chunks, model=self.mock_model)

        # Only c1 and c6 should be embedded
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertEqual(results[1]["chunk_id"], "c6")
        self.assertEqual(len(results[0]["embedding"]), EMBEDDING_DIMENSION)
        self.assertEqual(len(results[1]["embedding"]), EMBEDDING_DIMENSION)

    def test_all_invalid_chunks_returns_empty_list(self):
        """If no valid text is present across all chunks, returns empty list without calling model."""
        invalid_chunks = [
            {"chunk_id": "c1", "text": ""},
            {"chunk_id": "c2", "text": "   "},
            {"chunk_id": "c3"},
        ]
        results = generate_embeddings(invalid_chunks, model=self.mock_model)
        self.assertEqual(results, [])
        self.assertEqual(len(self.mock_model.encode_calls), 0)

    def test_input_immutability(self):
        """Input chunk dictionaries should not be mutated in-place."""
        original_chunk = {
            "chunk_id": "immutability_test",
            "text": "Testing that original dict is unchanged.",
        }
        chunks = [original_chunk]
        results = generate_embeddings(chunks, model=self.mock_model)

        self.assertNotIn("embedding", original_chunk)
        self.assertIn("embedding", results[0])

    def test_validation_dimension_mismatch_raises_error(self):
        """Raise ValueError if the model returns incorrect embedding dimensions."""
        wrong_dim_model = MockSentenceTransformer(dimension=128)

        with self.assertRaises(ValueError) as ctx:
            generate_embeddings(self.sample_chunks, model=wrong_dim_model)

        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_validation_count_mismatch_raises_error(self):
        """Raise ValueError if the model returns fewer vectors than expected."""
        broken_model = MagicMock()
        # Return only 1 vector when 2 chunks were sent
        broken_model.encode.return_value = np.zeros((1, EMBEDDING_DIMENSION))
        broken_model.get_sentence_embedding_dimension.return_value = EMBEDDING_DIMENSION

        with self.assertRaises(ValueError) as ctx:
            generate_embeddings(self.sample_chunks, model=broken_model)

        self.assertIn("count mismatch", str(ctx.exception).lower())

    def test_validation_nan_or_inf_raises_error(self):
        """Raise ValueError if vector contains NaN or Inf."""
        nan_vector = [0.1] * EMBEDDING_DIMENSION
        nan_vector[10] = float("nan")

        with self.assertRaises(ValueError) as ctx:
            validate_embeddings([nan_vector], expected_count=1, expected_dim=EMBEDDING_DIMENSION)
        self.assertIn("invalid non-finite float value", str(ctx.exception).lower())

        inf_vector = [0.1] * EMBEDDING_DIMENSION
        inf_vector[5] = float("inf")

        with self.assertRaises(ValueError) as ctx:
            validate_embeddings([inf_vector], expected_count=1, expected_dim=EMBEDDING_DIMENSION)
        self.assertIn("invalid non-finite float value", str(ctx.exception).lower())

    def test_cached_model_loader_is_lru_cached(self):
        """Verify load_embedding_model uses lru_cache."""
        self.assertTrue(hasattr(load_embedding_model, "cache_info"))

    def test_embed_query_success(self):
        """Verify embedding a single user question string."""
        vec = embed_query("What is the main topic?", model=self.mock_model)
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), EMBEDDING_DIMENSION)
        self.assertIsInstance(vec[0], float)

    def test_embed_query_empty_or_invalid_raises(self):
        """Verify invalid query strings raise ValueError."""
        with self.assertRaises(ValueError):
            embed_query("", model=self.mock_model)
        with self.assertRaises(ValueError):
            embed_query("   ", model=self.mock_model)
        with self.assertRaises(ValueError):
            embed_query(None, model=self.mock_model)

    def test_embed_query_dimension_mismatch_raises(self):
        """Verify ValueError is raised if query embedding has wrong dimension."""
        wrong_model = MockSentenceTransformer(dimension=128)
        with self.assertRaises(ValueError):
            embed_query("Valid query", model=wrong_model)


if __name__ == "__main__":
    unittest.main()
