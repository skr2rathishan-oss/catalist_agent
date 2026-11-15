"""Unit and integration tests for TranscriptParser and pipeline."""

import unittest
from pathlib import Path

from app.ingestion.loader import TranscriptLoader
from app.ingestion.parser import TranscriptParser


class TestTranscriptParser(unittest.TestCase):
    """Test suite for TranscriptParser."""

    def setUp(self):
        """Set up parser and test fixtures."""
        self.parser = TranscriptParser()

        # Sample clean transcript text simulating AMI format
        self.sample_raw_text = (
            "Meeting: ES2002a\n"
            "Team: ES2002\n"
            "Meeting number: 1 (a) of the team's four meetings\n"
            "Speakers: Speaker A, Speaker B, Speaker C, Speaker D\n"
            "Times: turn start time from the start of the recording (HH:MM:SS)\n"
            "Source: AMI Meeting Corpus, CC BY 4.0\n"
            "\n"
            "00:00:50 Speaker B\n"
            "Okay. Right. Um well this is the kick-off meeting for our project.\n"
            "\n"
            "00:01:06 Speaker D\n"
            "Mm-hmm. Great.\n"
        )

        self.sample_loaded = {
            "meeting_id": "ES2002a",
            "file_path": "data/raw/transcripts/ES2002a.txt",
            "raw_text": self.sample_raw_text,
        }

    def test_correct_meeting_metadata_extraction(self):
        """Test extraction of meeting_id, team_id, meeting_number, sequence."""
        result = self.parser.parse(self.sample_loaded)

        self.assertEqual(result["meeting_id"], "ES2002a")
        self.assertEqual(result["team_id"], "ES2002")
        self.assertEqual(result["meeting_number"], 1)
        self.assertEqual(result["meeting_sequence"], "a")
        self.assertEqual(result["filename"], "ES2002a.txt")

    def test_correct_speaker_id_extraction(self):
        """Test extraction of speakers list."""
        result = self.parser.parse(self.sample_loaded)

        self.assertEqual(result["speakers"], ["A", "B", "C", "D"])
        self.assertEqual(result["speaker_ids"], ["A", "B", "C", "D"])

    def test_correct_timestamp_extraction(self):
        """Test timestamp extraction from turn headers."""
        result = self.parser.parse(self.sample_loaded)
        utterances = result["utterances"]

        self.assertEqual(len(utterances), 2)
        self.assertEqual(utterances[0]["timestamp"], "00:00:50")
        self.assertEqual(utterances[1]["timestamp"], "00:01:06")

    def test_correct_utterance_separation(self):
        """Test separation of individual utterances and speaker assignment."""
        result = self.parser.parse(self.sample_loaded)
        utterances = result["utterances"]

        self.assertEqual(utterances[0]["utterance_id"], 1)
        self.assertEqual(utterances[0]["speaker_id"], "B")
        self.assertEqual(utterances[1]["utterance_id"], 2)
        self.assertEqual(utterances[1]["speaker_id"], "D")

    def test_character_offsets_exact_match(self):
        """Test that character offsets slice the exact original text byte-for-byte."""
        result = self.parser.parse(self.sample_loaded)
        raw = self.sample_loaded["raw_text"]

        for u in result["utterances"]:
            sliced = raw[u["start_char"]:u["end_char"]]
            self.assertEqual(sliced, u["original_text"])

    def test_multiline_utterance_handling(self):
        """Test that multi-line utterances are preserved without truncation."""
        multiline_text = (
            "Meeting: TS3001a\n"
            "Team: TS3001\n"
            "Meeting number: 1 (a)\n"
            "Speakers: Speaker A\n\n"
            "00:00:15 Speaker A\n"
            "This is line one of the utterance.\n"
            "This is line two of the same utterance.\n"
        )
        data = {"meeting_id": "TS3001a", "file_path": "test.txt", "raw_text": multiline_text}
        result = self.parser.parse(data)

        self.assertEqual(len(result["utterances"]), 1)
        u = result["utterances"][0]
        expected_text = "This is line one of the utterance.\nThis is line two of the same utterance."
        self.assertEqual(u["original_text"], expected_text)
        self.assertEqual(multiline_text[u["start_char"]:u["end_char"]], expected_text)

    def test_original_text_preservation(self):
        """Test that punctuation, disfluencies, and casing are not altered."""
        text = (
            "Meeting: TS3001a\n"
            "Team: TS3001\n"
            "Meeting number: 1 (a)\n"
            "Speakers: Speaker A\n\n"
            "00:00:10 Speaker A\n"
            "Um... well, we're gonna need L_C_D_ screens, right?! Yes.\n"
        )
        data = {"meeting_id": "TS3001a", "file_path": "test.txt", "raw_text": text}
        result = self.parser.parse(data)

        expected = "Um... well, we're gonna need L_C_D_ screens, right?! Yes."
        self.assertEqual(result["utterances"][0]["original_text"], expected)

    def test_empty_transcript_handling(self):
        """Test empty transcript string handling."""
        empty_data = {"meeting_id": "ES2002a", "file_path": "empty.txt", "raw_text": "   \n  \n"}
        result = self.parser.parse(empty_data)

        self.assertEqual(result["utterances"], [])
        self.assertEqual(result["meeting_id"], "ES2002a")
        self.assertEqual(result["team_id"], "ES2002")

    def test_missing_metadata_handling(self):
        """Test transcript missing standard headers."""
        raw_body_only = (
            "00:00:05 Speaker C\n"
            "We have no header lines in this file.\n"
        )
        data = {"meeting_id": "IS1002b", "file_path": "IS1002b.txt", "raw_text": raw_body_only}
        result = self.parser.parse(data)

        self.assertEqual(result["meeting_id"], "IS1002b")
        self.assertEqual(result["team_id"], "IS1002")
        self.assertEqual(result["meeting_sequence"], "b")
        self.assertEqual(result["meeting_number"], 2)
        self.assertEqual(len(result["utterances"]), 1)
        self.assertEqual(result["speakers"], ["C"])
        self.assertIn("speakers", result["missing_metadata"])

    def test_malformed_timestamp_handling(self):
        """Test turn header with malformed timestamp."""
        malformed_time_text = (
            "Meeting: TS3001a\n"
            "Team: TS3001\n"
            "Meeting number: 1 (a)\n"
            "Speakers: Speaker A\n\n"
            "00:15 Speaker A\n"
            "Speech with invalid two-part timestamp.\n"
        )
        data = {"meeting_id": "TS3001a", "file_path": "test.txt", "raw_text": malformed_time_text}
        result = self.parser.parse(data)

        self.assertEqual(len(result["utterances"]), 1)
        self.assertEqual(result["utterances"][0]["timestamp"], "00:15")
        self.assertEqual(result["utterances"][0]["speaker_id"], "A")
        self.assertTrue(len(result["malformed_entries"]) > 0)

    def test_unparsed_content_retention(self):
        """Test retention of unexpected content before first turn header."""
        unparsed_pre_text = (
            "Meeting: TS3001a\n"
            "Team: TS3001\n"
            "Meeting number: 1 (a)\n"
            "Speakers: Speaker A\n"
            "Times: HH:MM:SS\n"
            "Source: AMI\n\n"
            "NOTE: This line is unparsed notes.\n\n"
            "00:00:10 Speaker A\n"
            "Hello world.\n"
        )
        data = {"meeting_id": "TS3001a", "file_path": "test.txt", "raw_text": unparsed_pre_text}
        result = self.parser.parse(data)

        self.assertEqual(len(result["unparsed_content"]), 1)
        self.assertIn("NOTE: This line is unparsed notes.", result["unparsed_content"][0]["text"])

    def test_duplicate_meeting_id_detection(self):
        """Test detection and warning on duplicate meeting ID ingestion."""
        self.parser.reset()
        self.parser.parse(self.sample_loaded)
        second_result = self.parser.parse(self.sample_loaded)

        self.assertTrue(any("Duplicate meeting ID" in w for w in second_result["warnings"]))

    def test_filename_and_header_meeting_id_mismatch(self):
        """Test logging warning when filename ID differs from header ID."""
        mismatch_text = (
            "Meeting: TS9999a\n"
            "Team: TS9999\n"
            "Meeting number: 1 (a)\n"
            "Speakers: Speaker A\n\n"
            "00:00:01 Speaker A\n"
            "Hello.\n"
        )
        data = {"meeting_id": "ES2002a", "file_path": "ES2002a.txt", "raw_text": mismatch_text}
        with self.assertLogs("app.ingestion.parser", level="WARNING") as log_capture:
            result = self.parser.parse(data)
            self.assertTrue(any("Meeting ID mismatch" in msg for msg in log_capture.output))
        self.assertEqual(result["meeting_id"], "ES2002a")

    def test_invalid_input_handling(self):
        """Test error when input is not a dictionary."""
        with self.assertRaises(ValueError):
            self.parser.parse("not a dictionary")

    def test_integration_with_actual_es2002a(self):
        """Test complete loader to parser integration on actual ES2002a file."""
        if not Path("data/raw/transcripts/ES2002a.txt").is_file():
            self.skipTest("Local transcript ES2002a.txt is not available.")
        loader = TranscriptLoader(data_dir="data/raw/transcripts")
        loaded_file = loader.load_file(Path("data/raw/transcripts/ES2002a.txt"))

        self.assertIsNotNone(loaded_file)
        self.parser.reset()
        parsed = self.parser.parse(loaded_file)

        self.assertEqual(parsed["meeting_id"], "ES2002a")
        self.assertEqual(parsed["team_id"], "ES2002")
        self.assertEqual(parsed["meeting_number"], 1)
        self.assertEqual(parsed["meeting_sequence"], "a")
        self.assertEqual(parsed["speakers"], ["A", "B", "C", "D"])
        self.assertGreater(len(parsed["utterances"]), 0)

        first_turn = parsed["utterances"][0]
        self.assertEqual(first_turn["timestamp"], "00:00:50")
        self.assertEqual(first_turn["speaker_id"], "B")
        self.assertTrue(first_turn["original_text"].startswith("Okay. Right."))

        # Verify character offset slices match original text
        raw_text = parsed["original_transcript_text"]
        for u in parsed["utterances"]:
            self.assertEqual(raw_text[u["start_char"]:u["end_char"]], u["original_text"])


if __name__ == "__main__":
    unittest.main()
