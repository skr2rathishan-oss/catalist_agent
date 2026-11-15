"""Scoring must include failures and avoid pretending unjudged answers were scored."""

import unittest

from app.answer_evaluation import summarise_answers


class TestAnswerEvaluation(unittest.TestCase):
    def test_failures_remain_in_denominator(self):
        rows = [{"verdict": value} for value in ("correct", "partial", "no_answer", "error")]
        self.assertEqual(summarise_answers(rows)["answer_score"], 0.375)

    def test_unjudged_has_no_correctness_score(self):
        rows = [{"verdict": value} for value in ("correct", "not_judged")]
        self.assertIsNone(summarise_answers(rows)["answer_score"])


if __name__ == "__main__":
    unittest.main()
