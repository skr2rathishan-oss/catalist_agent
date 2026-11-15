"""Offline checks for the exact saved submission boundary."""

import json
from pathlib import Path
import tempfile
import unittest

from app.agents.brief_agent import BRIEF_SECTIONS
from app.submission import audit_submission


class TestSubmissionAudit(unittest.TestCase):
    def test_exact_quotes_and_team_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed = root / "parsed"
            parsed.mkdir()
            texts = {
                "ES2002a": "We agreed to use normal replaceable batteries for power.",
                "TS3012a": "We agreed to make a simple remote control design.",
            }
            for meeting_id, source in texts.items():
                (parsed / f"{meeting_id}.json").write_text(json.dumps({
                    "original_transcript_text": source,
                    "utterances": [{"original_text": source, "start_char": 0,
                                    "end_char": len(source)}],
                }), encoding="utf-8")
            questions = root / "questions.json"
            questions.write_text(json.dumps({"questions": [{"question_id": "Q1",
                                                               "team": "ES2002",
                                                               "question": "Which power source?"}]}),
                                 encoding="utf-8")
            output = root / "answers.json"
            brief = "\n\n".join(
                f'{heading}\nSource-backed point.\nEvidence: [TS3012a] "{texts["TS3012a"]}"'
                for heading, _ in BRIEF_SECTIONS)
            saved = {"answers": [{"question_id": "Q1", "answer": "Normal batteries.",
                                   "citations": [{"meeting_id": "ES2002a",
                                                  "quote": texts["ES2002a"]}]}],
                     "brief": brief}
            output.write_text(json.dumps(saved), encoding="utf-8")
            self.assertEqual(audit_submission(output, questions, parsed)["exact_quote_checks"], 5)
            saved["answers"][0]["citations"][0]["quote"] = "We agreed to use solar panels for power."
            output.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not exact"):
                audit_submission(output, questions, parsed)


if __name__ == "__main__":
    unittest.main()
