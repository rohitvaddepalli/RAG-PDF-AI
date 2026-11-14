"""
tests/test_chunker.py
---------------------
Unit tests for utils/chunker.py covering:
- Chunk size and overlap limits
- Independent page processing
- Preservation of page numbers and source filenames
- Deterministic, unique chunk IDs
- Word count and char count calculation
- Skipping empty and whitespace pages
- Structured dictionary output validation
"""

import unittest
from utils.chunker import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    chunk_pages,
    chunk_extracted_pdf,
    generate_chunk_id,
    get_text_splitter,
)


class TestPDFChunker(unittest.TestCase):

    def setUp(self):
        self.sample_filename = "machine_learning_notes.pdf"
        # 120 words (~720 characters) to ensure multiple chunks when size=700
        self.long_page_text = "Data science and machine learning power modern AI systems. " * 15

    def test_default_constants(self):
        """Verifies chunk size and overlap defaults meet specifications."""
        self.assertEqual(DEFAULT_CHUNK_SIZE, 700)
        self.assertEqual(DEFAULT_CHUNK_OVERLAP, 100)

    def test_invalid_chunk_overlap_raises_error(self):
        """Chunk overlap must be smaller than chunk size."""
        with self.assertRaises(ValueError):
            get_text_splitter(chunk_size=500, chunk_overlap=500)

    def test_structured_dict_keys(self):
        """Verifies each chunk returns a dictionary with all required keys."""
        pages = [
            {
                "page_number": 1,
                "text": "Antigravity agents and RAG pipelines.",
                "word_count": 5,
                "char_count": 37,
            }
        ]
        chunks = chunk_pages(pages, source_name=self.sample_filename)

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        expected_keys = {
            "chunk_id",
            "text",
            "page_number",
            "source",
            "chunk_index",
            "word_count",
            "char_count",
        }
        self.assertTrue(expected_keys.issubset(chunk.keys()))
        self.assertEqual(chunk["page_number"], 1)
        self.assertEqual(chunk["source"], self.sample_filename)
        self.assertEqual(chunk["chunk_index"], 0)
        self.assertEqual(chunk["word_count"], 5)
        self.assertEqual(chunk["char_count"], len(chunk["text"]))

    def test_process_pages_independently(self):
        """Verifies chunks do not cross page boundaries."""
        pages = [
            {"page_number": 1, "text": "Page one unique secret."},
            {"page_number": 2, "text": "Page two unique fact."},
        ]
        chunks = chunk_pages(pages, source_name=self.sample_filename)

        self.assertEqual(len(chunks), 2)
        # First chunk belongs strictly to page 1
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertIn("Page one", chunks[0]["text"])
        self.assertNotIn("Page two", chunks[0]["text"])

        # Second chunk belongs strictly to page 2
        self.assertEqual(chunks[1]["page_number"], 2)
        self.assertIn("Page two", chunks[1]["text"])
        self.assertNotIn("Page one", chunks[1]["text"])

    def test_skip_empty_and_whitespace_pages(self):
        """Verifies that completely blank or whitespace-only pages are skipped."""
        pages = [
            {"page_number": 1, "text": ""},
            {"page_number": 2, "text": "   \n\n  \t  "},
            {"page_number": 3, "text": "Real content on page 3"},
        ]
        chunks = chunk_pages(pages, source_name=self.sample_filename)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["page_number"], 3)
        self.assertEqual(chunks[0]["text"], "Real content on page 3")

    def test_deterministic_chunk_ids(self):
        """Verifies chunk IDs are reproducible, deterministic, and unique."""
        id1 = generate_chunk_id("report.pdf", page_number=1, chunk_index=0)
        id2 = generate_chunk_id("report.pdf", page_number=1, chunk_index=0)
        id3 = generate_chunk_id("report.pdf", page_number=1, chunk_index=1)

        self.assertEqual(id1, id2, "Chunk IDs must be deterministic across runs")
        self.assertNotEqual(id1, id3, "Different chunk indices must have distinct IDs")
        self.assertEqual(id1, "report.pdf_p1_c0")

    def test_chunking_with_overlap(self):
        """Verifies text exceeding chunk_size is split with overlap."""
        pages = [
            {"page_number": 1, "text": self.long_page_text}
        ]
        chunks = chunk_pages(
            pages,
            source_name=self.sample_filename,
            chunk_size=700,
            chunk_overlap=100,
        )

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk["char_count"], 700)
            self.assertEqual(chunk["word_count"], len(chunk["text"].split()))
            self.assertEqual(chunk["page_number"], 1)

    def test_chunk_extracted_pdf_convenience_function(self):
        """Verifies integration with the output schema of extract_text_from_pdf."""
        mock_pdf_data = {
            "file_name": "ai_handbook.pdf",
            "file_path": "uploads/ai_handbook.pdf",
            "page_count": 2,
            "total_words": 10,
            "has_extractable_text": True,
            "pages": [
                {"page_number": 1, "text": "Chapter 1: Foundations of RAG."},
                {"page_number": 2, "text": "Chapter 2: Vector Search & Embeddings."},
            ],
            "full_text": "Chapter 1: Foundations of RAG.\n\nChapter 2: Vector Search & Embeddings.",
        }

        chunks = chunk_extracted_pdf(mock_pdf_data)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["source"], "ai_handbook.pdf")
        self.assertEqual(chunks[0]["chunk_id"], "ai_handbook.pdf_p1_c0")
        self.assertEqual(chunks[1]["source"], "ai_handbook.pdf")
        self.assertEqual(chunks[1]["chunk_id"], "ai_handbook.pdf_p2_c0")


if __name__ == "__main__":
    unittest.main()
