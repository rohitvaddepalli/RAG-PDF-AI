import math
import os
import sqlite3
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
        self.store = VectorStore(db_path=":memory:", expected_dim=EMBEDDING_DIMENSION)

    def tearDown(self):
        self.store.close()

    def test_init_and_table_creation(self):
        self.assertEqual(self.store.count(), 0)

    def test_add_and_count_chunks(self):
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

    def test_prevent_duplicate_chunk_ids_with_upsert(self):
        initial_chunk = {
            "chunk_id": "unique_chunk_1",
            "text": "Original text content.",
            "source": "doc.pdf",
            "page_number": 1,
            "embedding": create_synthetic_vector(1.0),
        }
        self.store.add_chunks([initial_chunk])
        self.assertEqual(self.store.count(), 1)

        updated_chunk = {
            "chunk_id": "unique_chunk_1",
            "text": "Updated text content.",
            "source": "doc.pdf",
            "page_number": 1,
            "embedding": create_synthetic_vector(0.9),
        }
        self.store.add_chunks([updated_chunk], upsert=True)

        self.assertEqual(self.store.count(), 1)
        record = self.store.get_chunk("unique_chunk_1")
        self.assertIsNotNone(record)
        self.assertEqual(record["text"], "Updated text content.")

    def test_prevent_duplicate_chunk_ids_raises_when_no_upsert(self):
        chunk = {
            "chunk_id": "dup_chunk",
            "text": "Text content.",
            "source": "doc.pdf",
            "page_number": 1,
            "embedding": create_synthetic_vector(1.0),
        }
        self.store.add_chunks([chunk])

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.add_chunks([chunk], upsert=False)

    def test_cosine_similarity_exact_and_orthogonal(self):
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

    def test_top_k_ordering_descending(self):
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

        results = self.store.search(query_embedding=query, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["chunk_id"], "high")
        self.assertEqual(results[1]["chunk_id"], "mid")
        self.assertEqual(results[2]["chunk_id"], "low")
        self.assertGreaterEqual(results[0]["similarity"], results[1]["similarity"])
        self.assertGreaterEqual(results[1]["similarity"], results[2]["similarity"])

    def test_score_threshold_filtering(self):
        v1 = [0.0] * EMBEDDING_DIMENSION
        v1[0] = 1.0
        v2 = [0.0] * EMBEDDING_DIMENSION
        v2[1] = 1.0

        self.store.add_chunks([
            {"chunk_id": "c1", "text": "T1", "source": "s.pdf", "page_number": 1, "embedding": v1},
            {"chunk_id": "c2", "text": "T2", "source": "s.pdf", "page_number": 1, "embedding": v2},
        ])

        results = self.store.search(query_embedding=v1, top_k=5, score_threshold=0.5)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c1")

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

    def test_empty_inputs_handling(self):
        self.assertEqual(self.store.add_chunks([]), 0)
        self.assertEqual(self.store.add_chunks(None), 0)

        query = create_synthetic_vector(1.0)
        self.assertEqual(self.store.search(query_embedding=query), [])

        with self.assertRaises(ValueError):
            self.store.search(query_embedding=[])
        with self.assertRaises(ValueError):
            self.store.search(query_embedding=None)

    def test_invalid_dimension_handling(self):
        short_vec = [1.0, 0.0]

        with self.assertRaises(ValueError) as ctx:
            self.store.add_chunks([{
                "chunk_id": "short_chunk",
                "text": "Short",
                "source": "doc.pdf",
                "page_number": 1,
                "embedding": short_vec,
            }])
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

        with self.assertRaises(ValueError) as ctx:
            self.store.search(query_embedding=short_vec)
        self.assertIn("dimension mismatch", str(ctx.exception).lower())

    def test_zero_vector_handling(self):
        zero_vec = [0.0] * EMBEDDING_DIMENSION

        with self.assertRaises(ValueError) as ctx:
            self.store.search(query_embedding=zero_vec)
        self.assertIn("zero vector", str(ctx.exception).lower())

        self.store.add_chunks([{
            "chunk_id": "zero_chunk",
            "text": "Zero text",
            "source": "doc.pdf",
            "page_number": 1,
            "embedding": zero_vec,
        }])
        query = create_synthetic_vector(1.0)
        results = self.store.search(query_embedding=query, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0]["similarity"], 0.0)

    def test_non_finite_values_handling(self):
        nan_vec = [0.1] * EMBEDDING_DIMENSION
        nan_vec[10] = float("nan")

        with self.assertRaises(ValueError):
            self.store.add_chunks([{
                "chunk_id": "nan_chunk",
                "text": "NaN",
                "source": "doc.pdf",
                "page_number": 1,
                "embedding": nan_vec,
            }])

        with self.assertRaises(ValueError):
            self.store.search(query_embedding=nan_vec)

    def test_disk_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            disk_db_path = f.name

        try:
            with VectorStore(db_path=disk_db_path) as disk_store:
                disk_store.add_chunks([
                    {
                        "chunk_id": "persisted_chunk_1",
                        "text": "Persisted on disk.",
                        "source": "disk_test.pdf",
                        "page_number": 3,
                        "embedding": create_synthetic_vector(1.0),
                    }
                ])
                self.assertEqual(disk_store.count(), 1)

            with VectorStore(db_path=disk_db_path) as reloaded_store:
                self.assertEqual(reloaded_store.count(), 1)
                results = reloaded_store.search(create_synthetic_vector(1.0), top_k=1)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0]["chunk_id"], "persisted_chunk_1")
                self.assertEqual(results[0]["text"], "Persisted on disk.")
        finally:
            if os.path.exists(disk_db_path):
                os.remove(disk_db_path)

    def test_delete_and_clear(self):
        v = create_synthetic_vector(1.0)
        self.store.add_chunks([
            {"chunk_id": "c1", "text": "T1", "source": "s.pdf", "page_number": 1, "embedding": v},
            {"chunk_id": "c2", "text": "T2", "source": "s.pdf", "page_number": 2, "embedding": v},
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
