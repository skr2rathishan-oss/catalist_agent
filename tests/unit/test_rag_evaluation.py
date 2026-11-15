"""Offline tests for end-to-end RAG question loading, judging and scoring."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from app.rag_evaluation import (judge_answer, load_rag_questions, save_rag_report,
                                score_case, summarise)


class TestRagEvaluation(unittest.TestCase):
    def setUp(self):

        self.case = {"id": 1, "type": "factual", "question": "Q?", "expected_answer": "Batteries.",
                     "meetings": ["ES2002c", "ES2002d"], "team": "ES2002"}

    def test_question_teams_are_inferred(self):
        fixture = [
            {"id": 1, "question": "Q1?", "expected_answer": "A1", "source": ["ES2002a.txt"]},
            {"id": 2, "question": "Q2?", "expected_answer": "A2",
             "source": ["ES2002b.txt", "TS3003a.txt"]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            cases = {case["id"]: case for case in load_rag_questions(path)}
        self.assertEqual(cases[1]["team"], "ES2002")
        self.assertEqual(cases[1]["meetings"], ["ES2002a"])
        self.assertEqual(cases[2]["team"], "ALL")

    def test_invalid_files_are_rejected(self):

        valid = {"id": 1, "question": "Q?", "expected_answer": "A", "source": ["ES2002a.txt"]}
        for data in ([], [valid, valid], [dict(valid, source=["notes.txt"])], [dict(valid, expected_answer="")]):
            with self.subTest(data=data), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "q.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_rag_questions(path)

    def test_judge_output_is_validated(self):

        client = Mock()
        client.generate_json.return_value = {"verdict": "partial", "reason": "Missing price."}
        self.assertEqual(judge_answer(client, self.case, "Batteries"), ("partial", "Missing price."))
        payload = client.generate_json.call_args.args[1]
        self.assertEqual(payload["expected_answer"], "Batteries.")
        for bad in ({"verdict": "great", "reason": "x"}, {"verdict": "correct"}, ["correct"]):
            client.generate_json.return_value = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                judge_answer(client, self.case, "Batteries")

    def test_scoring_retrieval_citations_and_summary(self):

        answered = {"status": "answered", "answer": "Batteries.",
                    "citations": [{"meeting_id": "ES2002d", "quote": "q"}]}
        good = score_case(self.case, answered, ["ES2002d", "ES2002d", "TS3003a"], "correct", "ok", 3)
        self.assertEqual(good["retrieval_recall"], 0.5)
        self.assertTrue(good["citation_in_sources"])
        wrong_source = score_case(self.case, dict(answered, citations=[{"meeting_id": "TS3003a", "quote": "q"}]),
                                  [], "partial", "", 1)
        self.assertFalse(wrong_source["citation_in_sources"])
        failed = score_case(self.case, None, [], "error", None, 1, "GenerationError: x")
        summary = summarise([good, wrong_source, failed])
        self.assertAlmostEqual(summary["answer_score"], 0.5)
        self.assertEqual(summary["verdicts"]["correct"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertIsNone(summarise([score_case(self.case, answered, [], "not_judged", None, 1)])["answer_score"])

    def test_report_files_are_written(self):

        row = score_case(self.case, None, [], "no_answer", None, 1)
        report = {"question_file": "q.json", "method": "bm25", "model": "m", "judge": True,
                  "summary": summarise([row]), "questions": [row]}
        with tempfile.TemporaryDirectory() as directory:
            run_file, results = save_rag_report(report, Path(directory) / "runs", Path(directory) / "r.md")
            self.assertEqual(json.loads(run_file.read_text(encoding="utf-8"))["questions"][0]["id"], 1)
            self.assertIn("Q1 — no_answer", results.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
