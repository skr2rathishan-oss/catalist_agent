"""Offline checks for terminal pacing, independent agent inputs and limit stops."""

from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.generation import GenerationError
from scripts import test_rag


class TestRagTerminal(unittest.TestCase):
    def run_command(self, args, replies=None, answers=None):

        cases = [{"id": i, "question": f"ES2002 question {i}?", "team": "ES2002", "type": "factual",
                  "expected_answer": "PRIVATE REFERENCE", "meetings": ["ES2002a"]} for i in (1, 2)]
        agent = Mock()
        agent.answer.side_effect = answers or [{"status": "answered", "answer": "Candidate answer.",
                                              "citations": [{"meeting_id": "ES2002a", "quote": "Exact source quote here has six words."}],
                                              "trace": [{"round": 1}]}] * 2
        client = SimpleNamespace(api_requests=0, cache_hits=0)
        output = io.StringIO()
        with (patch("sys.argv", ["test_rag.py", "--method", "bm25", "--no-judge", *args]),
              patch.object(test_rag, "load_rag_questions", return_value=cases),
              patch.object(test_rag, "load_generation_settings", return_value=SimpleNamespace(model="test-model")),
              patch.object(test_rag, "create_client", return_value=client),
              patch.object(test_rag, "QuestionAnsweringAgent", return_value=agent),
              patch.object(test_rag, "save_rag_report", return_value=(Path("run.json"), Path("results.md"))) as save,
              patch("builtins.input", side_effect=replies or []), redirect_stdout(output)):
            try:
                test_rag.main()
                exit_code = 0
            except SystemExit as error:
                exit_code = error.code
        return agent, output.getvalue(), save, exit_code

    def test_one_question_prints_answer_reference_and_trace_without_gold_leak(self):

        agent, text, save, code = self.run_command(["--id", "1", "--show-trace", "--as-of", "a"])
        self.assertEqual(code, 0)
        agent.answer.assert_called_once_with("ES2002 question 1?", "ES2002", "a")
        self.assertIn("Candidate answer.", text)
        self.assertIn("PRIVATE REFERENCE", text)
        self.assertIn("TRACE:", text)
        self.assertEqual(len(save.call_args.args[0]["questions"]), 1)

    def test_interactive_skip_and_quit_make_no_model_requests(self):

        agent, text, save, code = self.run_command(["--interactive"], replies=["s", "q"])
        agent.answer.assert_not_called()
        save.assert_not_called()
        self.assertEqual(code, 0)
        self.assertIn("No questions were run", text)

    def test_interactive_enter_runs_only_the_confirmed_question(self):

        agent, text, save, code = self.run_command(["--interactive"], replies=["", "q"])
        self.assertEqual(agent.answer.call_count, 1)
        self.assertEqual(len(save.call_args.args[0]["questions"]), 1)

    def test_rate_limit_stops_remaining_questions_and_saves_partial_report(self):

        agent, text, save, code = self.run_command([], answers=[GenerationError("limited", status=429)])
        self.assertEqual(agent.answer.call_count, 1)
        self.assertEqual(code, 1)
        self.assertTrue(save.call_args.args[0]["stopped_for_limit"])
        self.assertIn("Change --model or wait", text)


if __name__ == "__main__":
    unittest.main()
