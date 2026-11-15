"""Offline tests for official-question loading, brief cutoffs and output shape."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from app.agents.brief_agent import BRIEF_SECTIONS, BriefAgent
from app.submission import load_submission_questions, write_submission


class TestSubmission(unittest.TestCase):
    def test_submission_questions_are_valid(self):
        fixture = {"questions": [
            {"question_id": "Q01", "team": "ES2002", "question": "What was decided?"},
            {"question_id": "Q02", "team": "TS3012", "question": "Who agreed?"},
        ]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            questions = load_submission_questions(path)
        self.assertEqual([item["question_id"] for item in questions], ["Q01", "Q02"])

    def test_brief_uses_four_grounded_sections_and_cutoff_c(self):

        qa = Mock()
        qa.answer.side_effect = [
            {"status": "answered", "answer": heading + " answer.",
             "citations": [{"meeting_id": "TS3012c", "quote": "six exact source words from meeting c."}]}
            for heading, _ in BRIEF_SECTIONS
        ]
        brief = BriefAgent(qa).build()
        self.assertEqual(qa.answer.call_count, 4)
        self.assertTrue(all(call.args[1:] == ("TS3012", "c") for call in qa.answer.call_args_list))
        self.assertEqual(brief.count("[TS3012c]"), 4)
        with self.assertRaises(ValueError):
            BriefAgent(qa).build(as_of="d")

    def test_submission_writer_keeps_only_public_schema(self):

        answer = {"question_id": "Q01", "answer": "A.",
                  "citations": [{"meeting_id": "ES2002a", "quote": "six exact source words are copied here."}]}
        with tempfile.TemporaryDirectory() as directory:
            path = write_submission([answer], "Brief", Path(directory) / "answers.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"answers": [answer], "brief": "Brief"})


if __name__ == "__main__":
    unittest.main()
