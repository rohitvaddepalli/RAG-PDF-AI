import math
import os
import tempfile
import unittest
import numpy as np

from utils.vector_store import VectorStore
from utils.embeddings import EMBEDDING_DIMENSION


def create_synthetic_vector(seed_val: float, dimension: int = EMBEDDING_DIMENSION) -> list[float]:
    vec = np.zeros(dimension, dtype=np.float32)
    vec[0] = seed_val
    vec[1] = 1.0 - abs(seed_val)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return [float(x) for x in vec]


class TestVectorStore(unittest.TestCase):

    def setUp(self):
        self.store = VectorStore(expected_dim=EMBEDDING_DIMENSION)

    def tearDown(self):
        self.store.close()

    def test_add_chunks_and_count(self):
        self.assertEqual(self.store.count(), 0)

        chunks = [
            {
                "chunk_id": "doc1.pdf_p1_c0",
                "text": "Introduction to AI agents.",
                "source": "doc1.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "word_count": 4,
                "char_count": 26,
                "embedding": create_synthetic_vector(1.0),
            },
            {
                "chunk_id": "doc1.pdf_p1_c1",
                "text": "Deep learning architectures.",
                "source": "doc1.pdf",
                "page_number": 1,
                "chunk_index": 1,
                "word_count": 3,
                "char_count": 28,
                "embedding": create_synthetic_vector(0.5),
            },
        ]
        stored_count = self.store.add_chunks(chunks)
        self.assertEqual(stored_count, 2)
        self.assertEqual(self.store.count(), 2)

    def test_similarity_ranking(self):
        vec_a = [0.0] * EMBEDDING_DIMENSION
        vec_a[0] = 1.0

        vec_b = [0.0] * EMBEDDING_DIMENSION
        vec_b[1] = 1.0

        chunks = [
            {
                "chunk_id": "c_a",
                "text": "Chunk A",
                "source": "math.pdf",
                "page_number": 1,
                "embedding": vec_a,
            },
            {
                "chunk_id": "c_b",
                "text": "Chunk B",
                "source": "math.pdf",
                "page_number": 1,
                "embedding": vec_b,
            },
        ]
        self.store.add_chunks(chunks)

        results = self.store.search(query_embedding=vec_a, top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c_a")
        self.assertAlmostEqual(results[0]["similarity"], 1.0, places=5)
        self.assertEqual(results[1]["chunk_id"], "c_b")
        self.assertAlmostEqual(results[1]["similarity"], 0.0, places=5)

    def test_top_k_behavior(self):
        v_high = [0.0] * EMBEDDING_DIMENSION
        v_high[0], v_high[1] = 0.9, 0.1

        v_mid = [0.0] * EMBEDDING_DIMENSION
        v_mid[0], v_mid[1] = 0.5, 0.5

        v_low = [0.0] * EMBEDDING_DIMENSION
        v_low[0], v_low[1] = 0.1, 0.9

        chunks = [
            {"chunk_id": "mid", "text": "Mid", "source": "s.pdf", "page_number": 1, "embedding": v_mid},
            {"chunk_id": "low", "text": "Low", "source": "s.pdf", "page_number": 1, "embedding": v_low},
            {"chunk_id": "high", "text": "High", "source": "s.pdf", "page_number": 1, "embedding": v_high},
        ]
        self.store.add_chunks(chunks)

        query = [0.0] * EMBEDDING_DIMENSION
        query[0] = 1.0

        res_1 = self.store.search(query_embedding=query, top_k=1)
        self.assertEqual(len(res_1), 1)
        self.assertEqual(res_1[0]["chunk_id"], "high")

        res_2 = self.store.search(query_embedding=query, top_k=2)
        self.assertEqual(len(res_2), 2)
        self.assertEqual(res_2[0]["chunk_id"], "high")
        self.assertEqual(res_2[1]["chunk_id"], "mid")

        res_5 = self.store.search(query_embedding=query, top_k=5)
        self.assertEqual(len(res_5), 3)

        res_zero = self.store.search(query_embedding=query, top_k=0)
        self.assertEqual(res_zero, [])

    def test_empty_store(self):
        query = create_synthetic_vector(1.0)
        self.assertEqual(self.store.search(query_embedding=query), [])
        self.assertEqual(self.store.count(), 0)

    def test_invalid_query_dimensions(self):
        short_vec = [1.0, 0.0]

        with self.assertRaises(ValueError) as ctx:
            self.store.search(query_embedding=short_vec)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

        with self.assertRaises(ValueError):
            self.store.search(query_embedding=[])

        with self.assertRaises(ValueError):
            self.store.search(query_embedding=None)

        with self.assertRaises(ValueError):
            self.store.add_chunks([{
                "chunk_id": "short_chunk",
                "text": "Short",
                "embedding": short_vec,
            }])

    def test_invalid_numerical_values(self):
        nan_vec = [0.1] * EMBEDDING_DIMENSION
        nan_vec[10] = float("nan")

        with self.assertRaises(ValueError):
            self.store.search(query_embedding=nan_vec)

        with self.assertRaises(ValueError):
            self.store.add_chunks([{
                "chunk_id": "nan_chunk",
                "text": "NaN",
                "embedding": nan_vec,
            }])

    def test_zero_vectors(self):
        zero_vec = [0.0] * EMBEDDING_DIMENSION

        with self.assertRaises(ValueError) as ctx:
            self.store.search(query_embedding=zero_vec)
        self.assertIn("zero vector", str(ctx.exception).lower())

        self.store.add_chunks([{
            "chunk_id": "zero_chunk",
            "text": "Zero text",
            "embedding": zero_vec,
        }])
        query = create_synthetic_vector(1.0)
        results = self.store.search(query_embedding=query, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["similarity"], 0.0)

    def test_save_and_load_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_json_path = f.name

        try:
            chunks = [
                {
                    "chunk_id": "json_chunk_1",
                    "text": "JSON persistence test.",
                    "source": "sample.pdf",
                    "page_number": 1,
                    "embedding": create_synthetic_vector(1.0),
                }
            ]
            self.store.add_chunks(chunks)
            self.store.save(temp_json_path)

            new_store = VectorStore(expected_dim=EMBEDDING_DIMENSION)
            new_store.load(temp_json_path)

            self.assertEqual(new_store.count(), 1)
            results = new_store.search(create_synthetic_vector(1.0), top_k=1)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["chunk_id"], "json_chunk_1")
            self.assertEqual(results[0]["text"], "JSON persistence test.")
            self.assertEqual(results[0]["source"], "sample.pdf")
            self.assertAlmostEqual(results[0]["similarity"], 1.0, places=5)
        finally:
            if os.path.exists(temp_json_path):
                os.remove(temp_json_path)

    def test_metadata_preservation(self):
        v = create_synthetic_vector(1.0)
        self.store.add_chunks([
            {
                "chunk_id": "meta_chunk",
                "text": "Detailed content text.",
                "source": "financial_report.pdf",
                "page_number": 42,
                "chunk_index": 7,
                "word_count": 3,
                "char_count": 22,
                "embedding": v,
            }
        ])

        results = self.store.search(query_embedding=v, top_k=1)
        self.assertEqual(len(results), 1)
        res = results[0]

        self.assertEqual(res["chunk_id"], "meta_chunk")
        self.assertEqual(res["text"], "Detailed content text.")
        self.assertEqual(res["source"], "financial_report.pdf")
        self.assertEqual(res["page_number"], 42)
        self.assertEqual(res["chunk_index"], 7)
        self.assertEqual(res["word_count"], 3)
        self.assertEqual(res["char_count"], 22)
        self.assertIn("similarity", res)
        self.assertIn("score", res)

    def test_upsert_behavior(self):
        initial = {
            "chunk_id": "chunk_1",
            "text": "Version 1",
            "embedding": create_synthetic_vector(1.0),
        }
        self.store.add_chunks([initial])
        self.assertEqual(self.store.count(), 1)

        updated = {
            "chunk_id": "chunk_1",
            "text": "Version 2",
            "embedding": create_synthetic_vector(0.8),
        }
        self.store.add_chunks([updated], upsert=True)
        self.assertEqual(self.store.count(), 1)
        self.assertEqual(self.store.get_chunk("chunk_1")["text"], "Version 2")

        with self.assertRaises(ValueError):
            self.store.add_chunks([updated], upsert=False)

    def test_delete_and_clear(self):
        v = create_synthetic_vector(1.0)
        self.store.add_chunks([
            {"chunk_id": "c1", "text": "T1", "embedding": v},
            {"chunk_id": "c2", "text": "T2", "embedding": v},
        ])
        self.assertEqual(self.store.count(), 2)

        deleted = self.store.delete_chunk("c1")
        self.assertTrue(deleted)
        self.assertEqual(self.store.count(), 1)
        self.assertIsNone(self.store.get_chunk("c1"))
        self.assertIsNotNone(self.store.get_chunk("c2"))

        self.store.clear()
        self.assertEqual(self.store.count(), 0)


if __name__ == "__main__":
    unittest.main()
