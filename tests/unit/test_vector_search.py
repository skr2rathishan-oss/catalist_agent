"""Test vector indexing and hybrid retrieval with deterministic offline vectors."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from app.index import build_keyword_index, update_keyword_index
from app.retriever import search_hybrid
from app.vector_store import build_vector_index, update_vector_index, search_vectors


class FakeEmbeddingClient:
    """Small vectors keep tests readable without pretending to measure quality."""

    def __init__(self):

        self.settings = SimpleNamespace(model="nvidia/nemotron-3.5-embed-8b", dimensions=3, batch_size=2)
        self.calls = []

    def embed(self, texts, task="document"):

        self.calls.append((texts, task))
        vectors = []
        for text in texts:
            if "solar" in text.lower() or "sunlight" in text.lower():
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([1.0, 0.0, 0.0])
        return vectors


class TestVectorSearch(unittest.TestCase):
    def setUp(self):

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.chunks = self.root / "chunks"
        self.chunks.mkdir()
        self.keyword_index = self.root / "bm25.sqlite3"
        self.vector_index = self.root / "vectors.sqlite3"
        self.client = FakeEmbeddingClient()
        self.write_meeting("ES2002a", "A battery powers the remote control.")
        self.write_meeting("ES2002b", "A solar cell is our proposed power source.")
        self.write_meeting("ES2002d", "The final design uses a solar cell.")
        self.write_meeting("TS3012a", "Solar solar solar for a different team.")
        build_keyword_index(self.chunks, self.keyword_index)
        self.counts = build_vector_index(self.client, self.keyword_index, self.vector_index)
        self.client.calls.clear()

    def write_meeting(self, meeting_id, text):

        chunk = {
            "chunk_id": meeting_id + "_c001", "meeting_id": meeting_id,
            "team_id": meeting_id[:-1], "meeting_number": "abcd".index(meeting_id[-1]) + 1,
            "meeting_sequence": meeting_id[-1], "original_text": text,
            "turn_ids": [1], "speaker_ids": ["A"],
            "start_timestamp": "00:00:01", "end_timestamp": "00:00:02",
        }
        meeting = {key: chunk[key] for key in ("meeting_id", "team_id", "meeting_number", "meeting_sequence")}
        meeting.update(total_chunks=1, chunks=[chunk])
        (self.chunks / f"{meeting_id}.json").write_text(json.dumps(meeting), encoding="utf-8")

    def search(self, query="sunlight", **kwargs):

        return search_vectors(query, "ES2002", client=self.client, index_path=self.vector_index, **kwargs)

    def test_semantic_match_preserves_source_metadata(self):

        self.assertEqual(self.counts["chunks"], 4)
        result = self.search(limit=1)[0]
        self.assertEqual(result["meeting_id"], "ES2002b")
        self.assertEqual(result["turn_ids"], [1])
        self.assertIn("solar cell", result["original_text"])
        self.assertAlmostEqual(result["score"], 1.0)
        self.assertEqual(self.client.calls[0][1], "query")

    def test_incremental_vector_update_embeds_only_changed_chunk(self):
        self.write_meeting("ES2002a", "The final choice uses solar power.")
        update_keyword_index(self.chunks, self.keyword_index)
        result = update_vector_index(self.client, self.keyword_index, self.vector_index)
        self.assertEqual(result["updated_chunks"], 1)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0][1], "document")
        self.client.calls.clear()
        self.assertEqual(update_vector_index(self.client, self.keyword_index, self.vector_index)["updated_chunks"], 0)
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.search(as_of="a", limit=1)[0]["meeting_id"], "ES2002a")

    def test_hybrid_exact_phase_does_not_inject_latest_meeting(self):

        results = search_hybrid("final power source", "ES2002", client=self.client,
                                keyword_index=self.keyword_index, vector_index=self.vector_index,
                                meeting_number=2)
        self.assertTrue(results)
        self.assertEqual({item["meeting_id"] for item in results}, {"ES2002b"})
        self.assertEqual(search_hybrid("power source", "ES2002", client=self.client,
                                      keyword_index=self.keyword_index, vector_index=self.vector_index,
                                      meeting_number=3), [])
        with self.assertRaises(ValueError):
            search_hybrid("power", "ES2002", as_of="a", meeting_number=2, client=self.client,
                          keyword_index=self.keyword_index, vector_index=self.vector_index)

    def test_team_and_cutoff_are_applied_before_ranking(self):

        results = self.search(as_of="c", limit=10)
        self.assertEqual({result["meeting_id"] for result in results}, {"ES2002a", "ES2002b"})
        self.assertTrue(all(result["team_id"] == "ES2002" for result in results))

    def test_optional_threshold(self):

        self.assertEqual([result["meeting_id"] for result in self.search(as_of="c", min_score=0.5)], ["ES2002b"])
        with self.assertRaises(ValueError):
            self.search(min_score=1.5)

    def test_no_query_call_for_empty_or_unknown_team(self):

        self.assertEqual(self.search(query=" "), [])
        self.assertEqual(search_vectors("sunlight", "TS9999", client=self.client, index_path=self.vector_index), [])
        self.assertEqual(self.client.calls, [])

    def test_model_or_dimension_mismatch_fails_before_call(self):

        self.client.settings.dimensions = 4
        with self.assertRaisesRegex(ValueError, "rebuild"):
            self.search()
        self.assertEqual(self.client.calls, [])

    def test_failed_build_keeps_old_vectors(self):

        original = self.client.embed

        def fail(texts, task="document"):

            raise RuntimeError("Provider unavailable")

        self.client.embed = fail
        with self.assertRaises(RuntimeError):
            build_vector_index(self.client, self.keyword_index, self.vector_index)
        self.client.embed = original
        self.assertEqual(self.search(limit=1)[0]["meeting_id"], "ES2002b")

    def test_hybrid_fuses_ranks_and_deduplicates(self):

        results = search_hybrid("solar", "ES2002", "c", 5, self.client, self.keyword_index, self.vector_index)
        ids = [result["chunk_id"] for result in results]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(results[0]["meeting_id"], "ES2002b")
        self.assertEqual(set(results[0]["retrieval_scores"]), {"bm25", "embedding"})
        self.assertTrue(all(result["meeting_number"] <= 3 for result in results))

    def test_hybrid_rejects_stale_corpus_before_api_call(self):

        self.write_meeting("ES2002b", "The design changed to a battery.")
        build_keyword_index(self.chunks, self.keyword_index)
        with self.assertRaisesRegex(ValueError, "Chunks changed"):
            search_hybrid("solar", "ES2002", client=self.client,
                          keyword_index=self.keyword_index, vector_index=self.vector_index)
        self.assertEqual(self.client.calls, [])

    def test_idempotent_vector_rebuild(self):

        before = self.search()
        build_vector_index(self.client, self.keyword_index, self.vector_index)
        self.assertEqual(self.search(), before)

    def test_exact_meeting_filter_and_cutoff_validation(self):

        results = self.search(meeting_number=2)
        self.assertEqual([item["meeting_id"] for item in results], ["ES2002b"])
        with self.assertRaises(ValueError):
            self.search(as_of="c", meeting_number=4)

    def test_final_search_uses_latest_available_meeting_below_cutoff(self):

        # This case needs decision evidence, not the default tentative proposal.
        self.write_meeting("ES2002b", "We decided on solar power instead of a battery.")
        build_keyword_index(self.chunks, self.keyword_index)
        build_vector_index(self.client, self.keyword_index, self.vector_index)
        self.client.calls.clear()
        results = search_hybrid("What power source did they finally choose?", "ES2002", "c", 5,
                                self.client, self.keyword_index, self.vector_index)
        self.assertEqual(results[0]["meeting_id"], "ES2002b")
        self.assertIn("Latest available meeting", results[0]["retrieval_reason"])
        self.assertNotIn("ES2002d", [item["meeting_id"] for item in results])

    def test_latest_proposal_is_not_promoted_as_decision_evidence(self):

        # The default latest eligible meeting contains only a solar proposal.
        results = search_hybrid("What power source did they finally choose?", "ES2002", "c", 5,
                                self.client, self.keyword_index, self.vector_index)
        self.assertTrue(results)
        self.assertIn("ES2002b", [item["meeting_id"] for item in results])
        self.assertFalse(any("Latest available meeting" in item["retrieval_reason"] for item in results))
        self.assertNotIn("ES2002d", [item["meeting_id"] for item in results])

    def test_historical_search_does_not_force_latest_meeting(self):

        results = search_hybrid("What battery did they initially propose?", "ES2002", "d", 5,
                                self.client, self.keyword_index, self.vector_index)
        self.assertTrue(results)
        self.assertFalse(any("Latest available meeting" in item["retrieval_reason"] for item in results))


if __name__ == "__main__":
    unittest.main()
