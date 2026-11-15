"""Unit tests for TranscriptChunker."""

from pathlib import Path
import unittest

from app.ingestion.chunker import TranscriptChunker


class TestTranscriptChunker(unittest.TestCase):
    """Test suite for TranscriptChunker class."""

    def test_zero_overlap(self):
        chunks = TranscriptChunker(overlap=0).chunk_meeting(self.sample_meeting)
        for left, right in zip(chunks, chunks[1:]):
            self.assertFalse(set(left["turn_ids"]) & set(right["turn_ids"]))

    def test_invalid_settings(self):
        for settings in ({"target_size": 0}, {"maximum_size": 100}, {"overlap": -1}, {"overlap": 400}):
            with self.assertRaises(ValueError):
                TranscriptChunker(**settings)

    def test_split_mode_preserves_source_spans(self):
        text = "Word " * 650
        meeting = {"meeting_id": "TS3012a", "team_id": "TS3012", "utterances": [
            {"utterance_id": 1, "speaker_id": "A", "timestamp": "00:00:01", "original_text": text, "start_char": 0, "end_char": len(text)}
        ]}
        chunks = TranscriptChunker(preserve_complete_utterances=False, overlap=0).chunk_meeting(meeting)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(sum(len(c["original_text"].split("\n", 1)[1].split()) for c in chunks), 650)
        for chunk in chunks:
            self.assertLessEqual(chunk["token_count"], 500)
            self.assertEqual(text[chunk["start_char"]:chunk["end_char"]], chunk["original_text"].split("\n", 1)[1])

    def test_batch_counts_successes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input"
            source.mkdir()
            (source / "broken.json").write_text("not json", encoding="utf-8")
            self.assertEqual(self.chunker.process_all(source, Path(directory) / "output"), 0)

    def setUp(self):
        """Set up chunker instance and sample parsed meeting fixtures."""
        self.chunker = TranscriptChunker(
            target_size=400,
            maximum_size=500,
            overlap=50,
            preserve_complete_utterances=True,
        )

        # Generate sample sequential utterances
        self.sample_utterances = []
        for i in range(1, 31):
            speaker = ["A", "B", "C", "D"][(i - 1) % 4]
            self.sample_utterances.append({
                "utterance_id": i,
                "timestamp": f"00:{i:02d}:00",
                "speaker_id": speaker,
                "original_text": f"This is utterance number {i} with several words for token accumulation. " * 3,
                "start_char": (i - 1) * 100,
                "end_char": i * 100 - 10,
            })

        self.sample_meeting = {
            "meeting_id": "TS3001a",
            "team_id": "TS3001",
            "meeting_number": 1,
            "meeting_sequence": "a",
            "utterances": self.sample_utterances,
        }

    def test_normal_chunk_creation(self):
        """Test normal chunk creation and unique chunk IDs."""
        chunks = self.chunker.chunk_meeting(self.sample_meeting)

        self.assertGreater(len(chunks), 1)
        # Verify chunk ID format
        self.assertEqual(chunks[0]["chunk_id"], "TS3001a_c001")
        self.assertEqual(chunks[1]["chunk_id"], "TS3001a_c002")

    def test_target_and_maximum_chunk_sizes(self):
        """Test that chunks conform to target and maximum token size bounds."""
        chunks = self.chunker.chunk_meeting(self.sample_meeting)

        for chunk in chunks:
            # Under standard conditions, chunk size should not exceed maximum size
            self.assertLessEqual(chunk["token_count"], self.chunker.maximum_size)

    def test_overlap_handling(self):
        """Test that consecutive chunks share overlapping turns."""
        chunks = self.chunker.chunk_meeting(self.sample_meeting)

        self.assertGreater(len(chunks), 1)
        for i in range(len(chunks) - 1):
            curr_turns = set(chunks[i]["turn_ids"])
            next_turns = set(chunks[i + 1]["turn_ids"])
            overlap_turns = curr_turns.intersection(next_turns)
            # Ensure adjacent chunks overlap by at least one turn
            self.assertGreater(len(overlap_turns), 0)

    def test_metadata_preservation(self):
        """Test preservation of required metadata fields in chunks."""
        chunks = self.chunker.chunk_meeting(self.sample_meeting)
        first_chunk = chunks[0]

        # Verify all required fields from configuration
        self.assertIn("chunk_id", first_chunk)
        self.assertEqual(first_chunk["meeting_id"], "TS3001a")
        self.assertEqual(first_chunk["team_id"], "TS3001")
        self.assertIn("turn_ids", first_chunk)
        self.assertIn("speaker_ids", first_chunk)
        self.assertIn("start_timestamp", first_chunk)
        self.assertIn("end_timestamp", first_chunk)
        self.assertIn("original_text", first_chunk)

    def test_original_text_preservation(self):
        """Test that original utterance text is preserved in chunk text."""
        chunks = self.chunker.chunk_meeting(self.sample_meeting)
        first_chunk = chunks[0]

        first_turn_id = first_chunk["turn_ids"][0]
        original_u = next(u for u in self.sample_utterances if u["utterance_id"] == first_turn_id)
        # Verify complete utterance text appears in chunk
        self.assertIn(original_u["original_text"], first_chunk["original_text"])

    def test_long_utterances(self):
        """Test handling of an utterance that exceeds maximum token size alone."""
        long_text = "Word " * 650
        long_meeting = {
            "meeting_id": "TS3099a",
            "team_id": "TS3099",
            "utterances": [
                {
                    "utterance_id": 1,
                    "timestamp": "00:00:10",
                    "speaker_id": "A",
                    "original_text": long_text,
                }
            ],
        }

        chunks = self.chunker.chunk_meeting(long_meeting)

        # Ensure chunk is created without error
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["turn_ids"], [1])
        self.assertGreater(chunks[0]["token_count"], self.chunker.maximum_size)

    def test_empty_meetings(self):
        """Test handling of meetings with no utterances."""
        empty_meeting = {
            "meeting_id": "EMPTY01a",
            "team_id": "EMPTY01",
            "utterances": [],
        }

        chunks = self.chunker.chunk_meeting(empty_meeting)
        self.assertEqual(chunks, [])

    def test_chronological_ordering(self):
        """Test that turns inside and across chunks are strictly chronological."""
        chunks = self.chunker.chunk_meeting(self.sample_meeting)

        for chunk in chunks:
            # Turn IDs within a chunk must be monotonically increasing
            self.assertEqual(chunk["turn_ids"], sorted(chunk["turn_ids"]))

    def test_chunks_reference_valid_original_turn_ids(self):
        """Test that all turn IDs in chunks exist in the source meeting."""
        chunks = self.chunker.chunk_meeting(self.sample_meeting)
        valid_ids = {u["utterance_id"] for u in self.sample_utterances}

        for chunk in chunks:
            for turn_id in chunk["turn_ids"]:
                self.assertIn(turn_id, valid_ids)

    def test_integration_with_actual_parsed_file(self):
        """Test chunking with actual parsed file ES2002a.json."""
        parsed_file = Path("data/parsed/ES2002a.json")
        if not parsed_file.exists():
            self.skipTest("data/parsed/ES2002a.json does not exist.")

        import json
        data = json.loads(parsed_file.read_text(encoding="utf-8"))
        chunks = self.chunker.chunk_meeting(data)

        self.assertGreater(len(chunks), 5)
        # Verify turn coverage across all chunks
        covered_turns = set()
        for c in chunks:
            covered_turns.update(c["turn_ids"])
        total_source_turns = {u["utterance_id"] for u in data["utterances"]}
        self.assertEqual(covered_turns, total_source_turns)


if __name__ == "__main__":
    unittest.main()

