"""Unit tests for TranscriptLoader using standard library unittest."""

import tempfile
import unittest
from pathlib import Path

from app.ingestion.loader import TranscriptLoader


class TestTranscriptLoader(unittest.TestCase):
    """Test suite for TranscriptLoader class."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_valid_file(self):
        sample_file = self.temp_path / "TS3012a.txt"
        sample_content = "Meeting: TS3012a\n00:00:01 Speaker A\nHello everyone.\n"
        sample_file.write_text(sample_content, encoding="utf-8")

        loader = TranscriptLoader(data_dir=self.temp_path)
        result = loader.load_file(sample_file)

        self.assertIsNotNone(result)
        self.assertEqual(result["meeting_id"], "TS3012a")
        self.assertEqual(result["raw_text"], sample_content)

    def test_skip_empty_file(self):
        empty_file = self.temp_path / "TS3012b.txt"
        empty_file.write_text("   \n  \n", encoding="utf-8")

        loader = TranscriptLoader(data_dir=self.temp_path)
        result = loader.load_file(empty_file)

        self.assertIsNone(result)

    def test_handle_missing_directory(self):
        non_existent_path = self.temp_path / "non_existent_folder"
        loader = TranscriptLoader(data_dir=non_existent_path)
        transcripts = loader.load_all()

        self.assertEqual(transcripts, [])

    def test_load_all_preserves_text_and_counts(self):
        for name in ("ES2002a.txt", "ES2002b.txt"):
            (self.temp_path / name).write_text(
                f"Meeting: {Path(name).stem}\n00:00:01 Speaker A\nHello everyone.\n",
                encoding="utf-8",
            )
        loader = TranscriptLoader(data_dir=self.temp_path)
        transcripts = loader.load_all()

        self.assertEqual(len(transcripts), 2)
        self.assertTrue(all("meeting_id" in t and "raw_text" in t for t in transcripts))


if __name__ == "__main__":
    unittest.main()

