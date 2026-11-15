"""Exercise a real temporary SQLite FTS5 index without network calls."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from app.index import build_keyword_index, update_keyword_index
from app.retriever import search_keywords


class TestKeywordSearch(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.chunks_dir = self.root / "chunks"
        self.chunks_dir.mkdir()
        self.index = self.root / "bm25.sqlite3"
        self.write_meeting("ES2002a", "The battery is one possible power source.")
        self.write_meeting("ES2002b", "Choose a lithium battery and an L_C_D_ screen.")
        self.write_meeting("ES2002c", "The wooden case has a comfortable shape.")
        self.write_meeting("ES2002d", "The final battery and L_C_D_ screen are approved.")
        self.write_meeting("TS3012d", "Lithium battery lithium battery lithium battery.")
        self.counts = build_keyword_index(self.chunks_dir, self.index)

    def write_meeting(self, meeting_id, text):
        number = "abcd".index(meeting_id[-1]) + 1
        chunk = {
            "chunk_id": meeting_id + "_c001", "meeting_id": meeting_id,
            "team_id": meeting_id[:-1], "meeting_number": number,
            "meeting_sequence": meeting_id[-1], "original_text": text,
            "turn_ids": [1], "speaker_ids": ["B"],
            "start_timestamp": "00:00:01", "end_timestamp": "00:00:02",
        }
        meeting = {key: chunk[key] for key in ("meeting_id", "team_id", "meeting_number", "meeting_sequence")}
        meeting.update(total_chunks=1, chunks=[chunk])
        path = self.chunks_dir / f"{meeting_id}.json"
        path.write_text(json.dumps(meeting), encoding="utf-8")
        return path

    def search(self, query, **kwargs):
        return search_keywords(query, "ES2002", index_path=self.index, **kwargs)

    def test_build_counts_and_ranking(self):
        self.assertEqual(self.counts, {"meetings": 5, "chunks": 5})
        results = self.search("lithium battery")
        self.assertEqual(results[0]["meeting_id"], "ES2002b")
        scores = [result["score"] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(score > 0 for score in scores))
        self.assertEqual(results[0]["_query_terms"], ["lithium", "battery"])

    def test_incremental_update_replaces_changed_meeting_and_removes_deleted_one(self):
        self.assertEqual(update_keyword_index(self.chunks_dir, self.index)["updated_meetings"], 0)
        self.write_meeting("ES2002b", "The new design uses a solar cell.")
        (self.chunks_dir / "ES2002c.json").unlink()
        counts = update_keyword_index(self.chunks_dir, self.index)
        self.assertEqual(counts["updated_meetings"], 2)
        self.assertEqual(counts["chunks"], 4)
        self.assertEqual([row["meeting_id"] for row in self.search("solar")], ["ES2002b"])
        self.assertNotIn("ES2002c", [row["meeting_id"] for row in self.search("wooden")])

    def test_team_filter_before_limit(self):
        results = self.search("lithium", limit=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["meeting_id"], "ES2002b")
        self.assertTrue(all(result["team_id"] == "ES2002" for result in self.search("battery")))

    def test_all_teams_searches_every_team_and_keeps_cutoff(self):
        results = search_keywords("lithium battery", "all", index_path=self.index)
        self.assertEqual({result["team_id"] for result in results}, {"ES2002", "TS3012"})
        self.assertEqual(results[0]["meeting_id"], "TS3012d")
        early = search_keywords("lithium battery", "ALL", as_of="b", index_path=self.index)
        self.assertEqual({result["meeting_id"] for result in early}, {"ES2002a", "ES2002b"})

    def test_explicit_question_team_overrides_all_before_ranking(self):
        results = search_keywords("ES2002 lithium battery", "all", index_path=self.index)
        self.assertTrue(results)
        self.assertTrue(all(result["team_id"] == "ES2002" for result in results))
        self.assertEqual(results[0]["meeting_id"], "ES2002b")

    def test_conflicting_question_and_selected_team_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Question names ES2002"):
            search_keywords("ES2002 lithium battery", "TS3012", index_path=self.index)

    def test_team_validation_rejects_other_values(self):
        from app.retriever import validate_search
        self.assertEqual(validate_search("q", " all ", "d", 5), "ALL")
        for team in ("ALLX", "", "ES20", None):
            with self.subTest(team=team), self.assertRaises(ValueError):
                validate_search("q", team, "d", 5)

    def test_inclusive_cutoff_before_limit(self):
        results = self.search("battery", as_of="a", limit=1)
        self.assertEqual([result["meeting_id"] for result in results], ["ES2002a"])
        results = self.search("battery LCD", as_of="c")
        self.assertTrue(results)
        self.assertTrue(all(result["meeting_number"] <= 3 for result in results))

    def test_unknown_team_returns_no_matches(self):
        self.assertEqual(search_keywords("battery", "ES9999", index_path=self.index), [])

    def test_acronym_normalisation_and_original_evidence(self):
        results = self.search("LCD")
        self.assertEqual({result["meeting_id"] for result in results}, {"ES2002b", "ES2002d"})
        self.assertEqual(self.search("L_C_D_"), results)
        for result in results:
            self.assertIn("L_C_D_", result["original_text"])
            self.assertIn("LCD", result["normalised_text"])
            self.assertEqual(result["turn_ids"], [1])
            self.assertEqual(result["speaker_ids"], ["B"])
            self.assertEqual(result["start_timestamp"], "00:00:01")

    def test_stemming(self):
        self.assertTrue(self.search("batteries"))

    def test_empty_queries_and_no_matches(self):
        for query in ("", "   ", "!!!", "What did the team do?", "ES2002"):
            self.assertEqual(self.search(query), [])
        self.assertEqual(self.search("unfindableword"), [])

    def test_fts_syntax_is_treated_as_words(self):
        self.assertEqual(self.search('battery" OR * )'), self.search("battery"))

    def test_invalid_arguments(self):
        for kwargs in ({"as_of": "z"}, {"as_of": ""}, {"limit": 0}, {"limit": True}, {"limit": 1.5}):
            with self.assertRaises(ValueError):
                self.search("battery", **kwargs)
        with self.assertRaises(ValueError):
            search_keywords("battery", "ES2002' OR 1=1", index_path=self.index)

    def test_missing_index_does_not_create_database(self):
        missing = self.root / "missing.sqlite3"
        with self.assertRaisesRegex(FileNotFoundError, "build.py"):
            search_keywords("battery", "ES2002", index_path=missing)
        self.assertFalse(missing.exists())

    def test_rebuild_is_idempotent_and_removes_old_entries(self):
        before = self.search("battery")
        self.assertEqual(build_keyword_index(self.chunks_dir, self.index), self.counts)
        self.assertEqual(self.search("battery"), before)
        self.write_meeting("ES2002a", "Only a wooden case remains.")
        build_keyword_index(self.chunks_dir, self.index)
        self.assertNotIn("ES2002a", [result["meeting_id"] for result in self.search("battery")])

    def test_failed_build_preserves_previous_index(self):
        before = self.search("battery")
        (self.chunks_dir / "broken.json").write_text("not JSON", encoding="utf-8")
        with self.assertRaises(ValueError):
            build_keyword_index(self.chunks_dir, self.index)
        self.assertEqual(self.search("battery"), before)

    def test_duplicate_chunks_are_rejected(self):
        path = self.chunks_dir / "ES2002a.json"
        (self.chunks_dir / "duplicate.json").write_bytes(path.read_bytes())
        with self.assertRaises(sqlite3.IntegrityError):
            build_keyword_index(self.chunks_dir, self.index)
        self.assertTrue(self.search("battery"))

    def test_inconsistent_metadata_is_rejected(self):
        path = self.chunks_dir / "ES2002a.json"
        meeting = json.loads(path.read_text(encoding="utf-8"))
        meeting["chunks"][0]["team_id"] = "TS3012"
        path.write_text(json.dumps(meeting), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata"):
            build_keyword_index(self.chunks_dir, self.index)

    def test_empty_input_preserves_previous_index(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "No chunk files"):
            build_keyword_index(empty, self.index)
        self.assertTrue(self.search("battery"))

    def test_exact_meeting_filter(self):
        self.assertEqual([item["meeting_id"] for item in self.search("battery", meeting_number=1)], ["ES2002a"])
        with self.assertRaises(ValueError):
            self.search("battery", as_of="a", meeting_number=2)


if __name__ == "__main__":
    unittest.main()
