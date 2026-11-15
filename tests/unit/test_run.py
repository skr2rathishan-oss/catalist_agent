"""Offline checks for generation diagnostics without retrieval or team inputs."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import unittest
from unittest.mock import Mock, patch

from app.config import GenerationSettings
from scripts import run


class TestRunGenerationCheck(unittest.TestCase):
    def test_check_accepts_explicit_model_without_running_retrieval(self):

        model = "nvidia/nemotron-3.5-lightning-30b-a3b"
        client = Mock(api_requests=1, cache_hits=0)
        client.available_models.return_value = [model]
        client.generate_json.return_value = {"ok": True}
        output = io.StringIO()
        with (patch("sys.argv", ["run.py", "--check-generation", "--generation-model", model]),
              patch.object(run, "load_generation_settings", return_value=GenerationSettings()),
              patch.object(run, "NvidiaGenerationClient", return_value=client) as factory,
              patch.object(run, "QuestionAnsweringAgent") as agent,
              patch.object(run, "NvidiaEmbeddingClient") as embeddings,
              redirect_stdout(output), redirect_stderr(io.StringIO())):
            run.main()
        self.assertEqual(factory.call_args.args[0].model, model)
        self.assertEqual(json.loads(output.getvalue())["generation_model"], model)
        client.generate_json.assert_called_once()
        agent.assert_not_called()
        embeddings.assert_not_called()

    def test_diagnostics_shows_loaded_paths_without_model_requests_or_key(self):

        model = "nvidia/nemotron-3.5-lightning-30b-a3b"
        output = io.StringIO()
        with (patch("sys.argv", ["run.py", "--diagnostics", "--generation-model", model]),
              patch.object(run, "load_generation_settings", return_value=GenerationSettings(api_key="private-test-key")),
              patch.object(run, "NvidiaGenerationClient") as factory,
              patch.object(run, "QuestionAnsweringAgent") as agent,
              patch.object(run, "NvidiaEmbeddingClient") as embeddings,
              redirect_stdout(output)):
            run.main()
        diagnostics = json.loads(output.getvalue())
        self.assertEqual(diagnostics["generation_model"], model)
        self.assertTrue(diagnostics["generation_module"].endswith("generation.py"))
        self.assertTrue(diagnostics["api_key_configured"])
        self.assertNotIn("private-test-key", output.getvalue())
        factory.assert_not_called()
        agent.assert_not_called()
        embeddings.assert_not_called()

    def test_absent_catalog_model_stops_before_generation(self):

        client = Mock()
        client.available_models.return_value = []
        errors = io.StringIO()
        with (patch("sys.argv", ["run.py", "--check-generation"]),
              patch.object(run, "load_generation_settings", return_value=GenerationSettings()),
              patch.object(run, "NvidiaGenerationClient", return_value=client),
              redirect_stderr(errors), self.assertRaises(SystemExit) as caught):
            run.main()
        self.assertEqual(caught.exception.code, 1)
        self.assertIn("absent from the authenticated catalog", errors.getvalue())
        client.generate_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
