import os
import tempfile
import unittest
import fitz
from utils.pdf_handler import extract_text_from_pdf, get_pdf_metadata


class TestPDFExtraction(unittest.TestCase):

    def test_normal_pdf_extraction(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            pdf_path = f.name

        try:
            doc = fitz.open()
            page1 = doc.new_page()
            page1.insert_text((50, 50), "Hello world from Page 1")

            page2 = doc.new_page()
            page2.insert_text((50, 50), "PyMuPDF extraction is awesome and fast")

            doc.save(pdf_path)
            doc.close()

            result = extract_text_from_pdf(pdf_path)

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["page_count"], 2)
            self.assertTrue(result["has_extractable_text"])

            self.assertEqual(result["pages"][0]["page_number"], 1)
            self.assertEqual(result["pages"][1]["page_number"], 2)

            self.assertEqual(result["pages"][0]["word_count"], 5)
            self.assertEqual(result["pages"][1]["word_count"], 6)
            self.assertEqual(result["total_words"], 11)

        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

    def test_empty_file_handling(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            empty_path = f.name

        try:
            result = extract_text_from_pdf(empty_path)
            self.assertEqual(result["status"], "error")
            self.assertIn("empty", result["error_message"].lower())
            self.assertEqual(result["page_count"], 0)
            self.assertEqual(result["pages"], [])
        finally:
            if os.path.exists(empty_path):
                os.remove(empty_path)

    def test_corrupted_file_handling(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"Not a valid PDF file contents")
            corrupt_path = f.name

        try:
            result = extract_text_from_pdf(corrupt_path)
            self.assertEqual(result["status"], "error")
            self.assertIn("corrupted or invalid", result["error_message"].lower())
            self.assertEqual(result["page_count"], 0)
        finally:
            if os.path.exists(corrupt_path):
                os.remove(corrupt_path)

    def test_no_extractable_text_detection(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            blank_path = f.name

        try:
            doc = fitz.open()
            doc.new_page()
            doc.save(blank_path)
            doc.close()

            result = extract_text_from_pdf(blank_path)
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["page_count"], 1)
            self.assertFalse(result["has_extractable_text"])
            self.assertEqual(result["total_words"], 0)
        finally:
            if os.path.exists(blank_path):
                os.remove(blank_path)


if __name__ == "__main__":
    unittest.main()
