import math
import unittest
from unittest.mock import MagicMock
from typing import Any, Dict, List, Union
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

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension: int = dimension
        self.encode_calls: List[Dict[str, Any]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        self.encode_calls.append({
            "texts": texts,
            "batch_size": batch_size,
            "show_progress_bar": show_progress_bar,
        })
        if isinstance(texts, str):
            return np.ones(self.dimension, dtype=np.float32) * 0.42
        return np.ones((len(texts), self.dimension), dtype=np.float32) * 0.42


class TestEmbeddingsModule(unittest.TestCase):

    def setUp(self) -> None:
        self.mock_model = MockSentenceTransformer()
        self.sample_chunks: List[Dict[str, Any]] = [
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

    def test_valid_input(self) -> None:
        results = generate_embeddings(self.sample_chunks, model=self.mock_model)

        self.assertEqual(len(results), 2)
        for chunk in results:
            self.assertIn("embedding", chunk)
            self.assertIsInstance(chunk["embedding"], list)
            self.assertEqual(len(chunk["embedding"]), EMBEDDING_DIMENSION)
            self.assertTrue(all(isinstance(x, float) for x in chunk["embedding"]))

    def test_empty_input(self) -> None:
        self.assertEqual(generate_embeddings([], model=self.mock_model), [])
        self.assertEqual(generate_embeddings(None, model=self.mock_model), [])

    def test_metadata_preservation(self) -> None:
        original_chunk = {
            "chunk_id": "c_orig",
            "text": "Metadata preservation test chunk.",
            "source": "test.pdf",
            "page_number": 3,
            "word_count": 4,
            "char_count": 33,
        }
        input_chunks = [dict(original_chunk)]

        results = generate_embeddings(input_chunks, model=self.mock_model)

        self.assertEqual(len(results), 1)
        res = results[0]
        for key in ["chunk_id", "text", "source", "page_number", "word_count", "char_count"]:
            self.assertEqual(res[key], original_chunk[key])

        self.assertNotIn("embedding", input_chunks[0])
        self.assertIn("embedding", res)

    def test_batch_encoding_is_utilized(self) -> None:
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

    def test_missing_and_invalid_text_handling(self) -> None:
        mixed_chunks = [
            {"chunk_id": "c1", "text": "Valid text here.", "page_number": 1},
            {"chunk_id": "c2", "text": "", "page_number": 1},
            {"chunk_id": "c3", "text": "   ", "page_number": 1},
            {"chunk_id": "c4", "page_number": 1},
            {"chunk_id": "c5", "text": None, "page_number": 1},
            "not a dictionary",
            {"chunk_id": "c6", "text": "Another valid chunk.", "page_number": 2},
        ]

        results = generate_embeddings(mixed_chunks, model=self.mock_model)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertEqual(results[1]["chunk_id"], "c6")
        self.assertEqual(len(results[0]["embedding"]), EMBEDDING_DIMENSION)
        self.assertEqual(len(results[1]["embedding"]), EMBEDDING_DIMENSION)

    def test_all_invalid_chunks_returns_empty_list(self) -> None:
        invalid_chunks = [
            {"chunk_id": "c1", "text": ""},
            {"chunk_id": "c2", "text": "   "},
            {"chunk_id": "c3"},
        ]
        results = generate_embeddings(invalid_chunks, model=self.mock_model)
        self.assertEqual(results, [])
        self.assertEqual(len(self.mock_model.encode_calls), 0)

    def test_incorrect_dimensions(self) -> None:
        wrong_dim_model = MockSentenceTransformer(dimension=128)

        with self.assertRaises(ValueError) as ctx:
            generate_embeddings(self.sample_chunks, model=wrong_dim_model)

        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_count_mismatch(self) -> None:
        broken_model = MagicMock()
        broken_model.encode.return_value = np.zeros((1, EMBEDDING_DIMENSION))
        broken_model.get_sentence_embedding_dimension.return_value = EMBEDDING_DIMENSION

        with self.assertRaises(ValueError) as ctx:
            generate_embeddings(self.sample_chunks, model=broken_model)

        self.assertIn("count mismatch", str(ctx.exception).lower())

    def test_nan_and_infinity(self) -> None:
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

    def test_model_caching(self) -> None:
        self.assertTrue(hasattr(load_embedding_model, "cache_info"))

    def test_embed_query_valid(self) -> None:
        vec = embed_query("What is machine learning?", model=self.mock_model)
        self.assertIsInstance(vec, list)
        self.assertEqual(len(vec), EMBEDDING_DIMENSION)
        self.assertTrue(all(isinstance(x, float) for x in vec))

    def test_embed_query_empty_or_invalid(self) -> None:
        with self.assertRaises(ValueError):
            embed_query("", model=self.mock_model)
        with self.assertRaises(ValueError):
            embed_query("   ", model=self.mock_model)
        with self.assertRaises(ValueError):
            embed_query(None, model=self.mock_model)

    def test_embed_query_incorrect_dimension(self) -> None:
        wrong_model = MockSentenceTransformer(dimension=128)
        with self.assertRaises(ValueError):
            embed_query("Valid query", model=wrong_model)


if __name__ == "__main__":
    unittest.main()
