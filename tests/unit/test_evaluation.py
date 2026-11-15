"""Offline tests for evidence coverage, honest reporting, and gold validation."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from app.evaluation import evaluate_retrieval, load_checked_questions, save_retrieval_report, score_evidence


class TestRetrievalEvaluation(unittest.TestCase):
    def setUp(self):

        self.quote = "We agreed to use normal batteries only."
        self.question = {
            "question_id": "T01", "question": "What power source was agreed?",
            "team": "ES2002", "as_of": "c", "split": "holdout", "checked_answer": "Normal batteries.",
            "evidence": [{"meeting_id": "ES2002b", "turn_id": 1, "quote": self.quote},
                         {"meeting_id": "ES2002c", "turn_id": 2, "quote": "No kinetic energy will be included here."}],
        }
        self.result = {"chunk_id": "ES2002b_c001", "meeting_id": "ES2002b", "team_id": "ES2002",
                       "meeting_number": 2, "turn_ids": [1], "original_text": self.quote}

    def test_partial_multimeeting_evidence_is_not_complete(self):

        score = score_evidence(self.question, [self.result])
        self.assertTrue(score["any_evidence_found"])
        self.assertFalse(score["all_evidence_found"])
        self.assertEqual(score["evidence_recall"], 0.5)
        self.assertEqual(score["evidence_ranks"], [1, None])

    def test_duplicate_chunks_cannot_inflate_recall(self):

        score = score_evidence(self.question, [self.result, self.result])
        self.assertEqual(score["evidence_recall"], 0.5)

    def test_complete_evidence_and_rank(self):

        second = dict(self.result, chunk_id="ES2002c_c001", meeting_id="ES2002c", meeting_number=3,
                      turn_ids=[2], original_text=self.question["evidence"][1]["quote"])
        irrelevant = dict(self.result, turn_ids=[99], original_text="Unrelated discussion")
        score = score_evidence(self.question, [irrelevant, self.result, second])
        self.assertTrue(score["all_evidence_found"])
        self.assertEqual(score["reciprocal_rank"], 0.5)

    def test_wrong_team_future_and_missing_quote_fail(self):

        results = [dict(self.result, team_id="TS3012"), dict(self.result, meeting_number=4),
                   dict(self.result, original_text="Paraphrased evidence")]
        score = score_evidence(self.question, results)
        self.assertFalse(score["any_evidence_found"])
        self.assertEqual(score["filter_violations"], 2)

    def test_search_does_not_receive_gold_answers(self):

        search = Mock(return_value=[self.result])
        report = evaluate_retrieval([self.question], search, 5)
        search.assert_called_once_with(self.question["question"], "ES2002", as_of="c", limit=5)
        self.assertEqual(report["splits"]["holdout"]["questions"], 1)
        self.assertEqual(report["summary"]["mean_evidence_recall"], 0.5)

    def test_failed_search_is_reported_and_counted(self):

        report = evaluate_retrieval([self.question], Mock(side_effect=RuntimeError("Unavailable")))
        self.assertEqual(report["summary"]["errors"], 1)
        self.assertEqual(report["summary"]["hit_rate"], 0)
        self.assertEqual(report["summary"]["questions"], 1)

    def test_loader_checks_original_turn_and_rejects_bad_gold(self):

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            question = dict(self.question, evidence=[self.question["evidence"][0]])
            source = {"original_transcript_text": self.quote,
                      "utterances": [{"utterance_id": 1, "original_text": self.quote,
                                      "start_char": 0, "end_char": len(self.quote)}]}
            (root / "ES2002b.json").write_text(json.dumps(source), encoding="utf-8")
            questions = root / "questions.json"
            questions.write_text(json.dumps({"questions": [question]}), encoding="utf-8-sig")
            self.assertEqual(len(load_checked_questions(questions, root)), 1)
            question["evidence"] = [dict(question["evidence"][0], quote="An invented quote that is not present.")]
            questions.write_text(json.dumps({"questions": [question]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Gold quote"):
                load_checked_questions(questions, root)

    def test_reports_preserve_runs_and_explain_scope(self):

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = {"methods": {"bm25": evaluate_retrieval([self.question], Mock(return_value=[]))}}
            first = save_retrieval_report(report, root / "runs", root / "results.md")
            second = save_retrieval_report(report, root / "runs", root / "results.md")
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_file())
            content = (root / "results.md").read_text(encoding="utf-8")
            self.assertIn("not generated-answer correctness", content)
            self.assertIn("T01", content)
