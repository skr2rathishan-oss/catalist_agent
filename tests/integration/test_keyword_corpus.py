"""Check BM25 against saved real chunks without API calls or persistent writes."""

import json
from pathlib import Path
import tempfile
import unittest

from app.config import CHUNKS_DIR, PARSED_DIR
from app.index import build_keyword_index
from app.retriever import search_keywords


class TestKeywordCorpus(unittest.TestCase):
    def test_real_evidence_and_brief_cutoff(self):
        if not (CHUNKS_DIR / "TS3012c.json").is_file() or not (PARSED_DIR / "TS3012c.json").is_file():
            self.skipTest("Run ingestion to create real parsed meetings and chunks first")
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / "bm25.sqlite3"
            counts = build_keyword_index(CHUNKS_DIR, index)
            self.assertEqual(counts["meetings"], 138)
            results = search_keywords("battery kinetic solar", "TS3012", as_of="c", index_path=index)
            self.assertTrue(results)
            for result in results:
                self.assertEqual(result["team_id"], "TS3012")
                self.assertIn(result["meeting_sequence"], "abc")
                parsed = json.loads((PARSED_DIR / f"{result['meeting_id']}.json").read_text(encoding="utf-8"))
                turns = {turn["utterance_id"]: turn for turn in parsed["utterances"]}
                for turn_id in result["turn_ids"]:
                    self.assertIn(turn_id, turns)
                    self.assertIn(turns[turn_id]["original_text"], result["original_text"])
            self.assertTrue(search_keywords("battery", "ES2002", index_path=index))


if __name__ == "__main__":
    unittest.main()
