"""Tests for search copies and a complete temporary ingestion run."""

import json
from pathlib import Path
import tempfile
import unittest

from app.config import PROJECT_ROOT, RAW_DIR, project_path
from app.ingest import build_ingestion, normalise_text


class TestIngest(unittest.TestCase):
    def test_search_copy(self):
        original = "Um we need twenty-five L_C_D_ screens. No."
        result = normalise_text(original)
        self.assertIn("25 LCD screens. No.", result)
        self.assertEqual(original, "Um we need twenty-five L_C_D_ screens. No.")

    def test_numbers(self):
        self.assertEqual(normalise_text("twelve Euro fifty"), "12.50 EUR")
        self.assertEqual(normalise_text("twelve and a half"), "12.5")
        self.assertEqual(normalise_text("twelve point five"), "12.5")
        self.assertEqual(normalise_text("twelve fifty"), "twelve fifty")

    def test_paths(self):
        self.assertEqual(project_path("data/raw/transcripts"), RAW_DIR)
        self.assertEqual(project_path(PROJECT_ROOT), PROJECT_ROOT)

    def test_temporary_pipeline_and_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            source = (
                "Meeting: TS3012a\nTeam: TS3012\nMeeting number: 1 (a)\n"
                "Speakers: Speaker A\n\n00:00:01 Speaker A\n"
                "Um we should keep the L_C_D_ screen for everyone.\n"
            )
            path = raw / "TS3012a.txt"
            path.write_text(source, encoding="utf-8")
            args = (raw, root / "parsed", root / "chunks")
            first = build_ingestion(*args)
            self.assertEqual(build_ingestion(*args), first)
            self.assertEqual(first, {"meetings": 1, "utterances": 1, "chunks": 1})
            self.assertEqual(path.read_text(encoding="utf-8"), source)
            meeting = json.loads((root / "parsed" / "TS3012a.json").read_text(encoding="utf-8"))
            turn = meeting["utterances"][0]
            self.assertIn("L_C_D_", turn["original_text"])
            self.assertIn("LCD", turn["normalised_text"])
            self.assertEqual(meeting["original_transcript_text"][turn["start_char"]:turn["end_char"]], turn["original_text"])
            incremental = build_ingestion(*args, incremental=True)
            self.assertEqual(incremental, {"meetings": 0, "utterances": 0, "chunks": 0, "skipped": 1})

    def test_missing_input_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                build_ingestion(Path(directory) / "missing")
