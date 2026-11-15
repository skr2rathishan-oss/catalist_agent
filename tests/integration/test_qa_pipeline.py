"""Real SQLite retrieval and generation cache with a fake HTTP provider, offline."""

from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from app.agents.prompts import QA_PLAN, QA_COVERAGE, QA_DRAFT, QA_REVIEW
from app.agents.qa_agent import QuestionAnsweringAgent
from app.config import GenerationSettings
from app.generation import NvidiaGenerationClient
from app.index import build_keyword_index
from app.retriever import search_keywords


class TestQuestionAnsweringPipeline(unittest.TestCase):
    def test_bm25_to_verified_answer_then_cached_offline_rerun(self):

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks, parsed = root / "chunks", root / "parsed"
            chunks.mkdir()
            parsed.mkdir()
            quote = "We agreed the power source will use batteries only."
            metadata = {"meeting_id": "ES2002b", "team_id": "ES2002", "meeting_sequence": "b", "meeting_number": 2}
            turn = {"utterance_id": 1, "speaker_id": "B", "timestamp": "00:00:01", "original_text": quote,
                    "start_char": 0, "end_char": len(quote)}
            (parsed / "ES2002b.json").write_text(json.dumps(dict(metadata, original_transcript_text=quote,
                                                               utterances=[turn])), encoding="utf-8")
            chunk = dict(metadata, chunk_id="ES2002b_c001", original_text=quote, turn_ids=[1],
                         speaker_ids=["B"], start_timestamp="00:00:01", end_timestamp="00:00:01")
            (chunks / "ES2002b.json").write_text(json.dumps(dict(metadata, total_chunks=1, chunks=[chunk])), encoding="utf-8")
            index = root / "bm25.sqlite3"
            build_keyword_index(chunks, index)
            citation = {"citation_id": "ES2002b_t1_w0"}
            outputs = {
                QA_PLAN: {"parts": [{"question": "What power source?", "query": "power source batteries",
                                     "requirements": ["What source was agreed?"]}]},
                QA_COVERAGE: {"parts": [{"id": "p1", "requirements": [
                    {"id": "p1_r1", "evidence": [citation]}], "next_query": None}]},
                QA_DRAFT: {"answer": "The team agreed to use batteries only.",
                           "citations": [dict(citation, part_ids=["p1"])]},
                QA_REVIEW: {"approved": True, "issues": []},
            }

            def transport(request, timeout):

                body = json.loads(request.data)
                output = outputs[body["messages"][0]["content"]]
                return io.BytesIO(json.dumps({"choices": [{"finish_reason": "stop",
                                                          "message": {"content": json.dumps(output)}}]}).encode())

            network = Mock(side_effect=transport)
            settings = GenerationSettings(api_key="offline-key")
            client = NvidiaGenerationClient(settings, root / "cache.sqlite3", root / "usage.jsonl", network, Mock())

            def search(query, team_id, **filters):

                return search_keywords(query, team_id, index_path=index, **filters)

            agent = QuestionAnsweringAgent(client, search, parsed)
            first = agent.answer("What power source was agreed in the functional design meeting?", "ES2002", "c")
            self.assertEqual(first["status"], "answered")
            self.assertEqual(first["citations"], [{"meeting_id": "ES2002b", "quote": quote}])
            self.assertEqual(network.call_count, 4)
            network.side_effect = AssertionError("A cached rerun must not make a network call")
            cached = NvidiaGenerationClient(replace(settings, api_key=""), root / "cache.sqlite3",
                                            root / "usage.jsonl", network, Mock())
            second = QuestionAnsweringAgent(cached, search, parsed).answer(
                "What power source was agreed in the functional design meeting?", "ES2002", "c")
            self.assertEqual(first, second)
            self.assertEqual(cached.cache_hits, 4)


if __name__ == "__main__":
    unittest.main()
