"""Real-corpus ES2002 regression; fake vectors isolate retrieval policy from APIs."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest

from app.config import CHUNKS_DIR, PARSED_DIR
from app.index import build_keyword_index
from app.retriever import search_hybrid, search_keywords
from app.agents.qa_agent import QuestionAnsweringAgent
from app.vector_store import build_vector_index


class ConstantEmbeddings:
    """Neutral scores let the test verify policy rather than embedding quality."""

    settings = SimpleNamespace(model="offline-policy-test", dimensions=2, batch_size=16)

    def embed(self, texts, task="document"):
        return [[1.0, 0.0] for text in texts]


class TestTemporalRetrieval(unittest.TestCase):
    def test_q5_focused_application_query_reaches_model_context(self):
        """Check source availability and admission, not live model answer quality."""

        if not all(path.exists() for path in (CHUNKS_DIR / "ES2002c.json", PARSED_DIR / "ES2002c.json")):
            self.skipTest("Run ingestion first")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks"
            chunks.mkdir()
            for file in CHUNKS_DIR.glob("ES2002?.json"):
                shutil.copyfile(file, chunks / file.name)
            index = root / "keywords.sqlite3"
            build_keyword_index(chunks, index)
            results = search_keywords("curvy texture colour", "ES2002", index_path=index, meeting_number=3)
            self.assertTrue(results)
            self.assertEqual({chunk["meeting_id"] for chunk in results}, {"ES2002c"})
            # A fixture assertion only: no gold words/turn IDs enter production retrieval.
            parsed = json.loads((PARSED_DIR / "ES2002c.json").read_text(encoding="utf-8"))
            gold = {turn["utterance_id"] for turn in parsed["utterances"]
                    if "very sort of curvy type shape" in turn["original_text"]}
            self.assertTrue(gold, "Checked Q5 application evidence is missing")
            agent = QuestionAnsweringAgent(client=object(), parsed_dir=PARSED_DIR)
            sources, meetings, admission = {}, {}, {}
            agent.add_sources(results, meetings, sources, "ES2002", "d", meeting_number=3, diagnostics=admission)
            self.assertTrue(gold & {turn_id for meeting_id, turn_id in sources},
                            f"Application evidence did not reach the model: {admission}")

    def test_es2002_final_power_choice_is_in_top_five(self):

        if not (CHUNKS_DIR / "ES2002d.json").exists():
            self.skipTest("Run ingestion first")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = root / "chunks"
            chunks.mkdir()
            for file in CHUNKS_DIR.glob("ES2002?.json"):
                shutil.copyfile(file, chunks / file.name)
            keywords, vectors = root / "keywords.sqlite3", root / "vectors.sqlite3"
            client = ConstantEmbeddings()
            build_keyword_index(chunks, keywords)
            build_vector_index(client, keywords, vectors)
            results = search_hybrid("What power source did the team finally choose?", "ES2002", "d", 5,
                                    client, keywords, vectors)
            parsed = json.loads((PARSED_DIR / "ES2002d.json").read_text(encoding="utf-8"))
            gold_turns = [turn["utterance_id"] for turn in parsed["utterances"]
                          if "No. Okay, just battery." in turn["original_text"]]
            self.assertTrue(gold_turns, "Checked source evidence is missing from the fixture")
            self.assertTrue(any(item["meeting_id"] == "ES2002d" and set(gold_turns) & set(item["turn_ids"])
                                for item in results), "Top five missed the actual final power decision")
            self.assertTrue(all(item["team_id"] == "ES2002" for item in results))
            self.assertTrue(any(item["meeting_number"] < 4 for item in results))
