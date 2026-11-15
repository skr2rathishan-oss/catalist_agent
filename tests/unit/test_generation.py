"""Offline tests for bounded generation, request isolation, and cache lifetime."""

from dataclasses import replace
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from app.config import GenerationSettings, load_generation_settings
from app.generation import GenerationError, NvidiaGenerationClient


def response(content='{"ok":true}', finish="stop"):

    return io.BytesIO(json.dumps({"choices": [{"finish_reason": finish, "message": {"content": content}}],
                                 "usage": {"prompt_tokens": 12, "completion_tokens": 4, "secret": "omit"}}).encode())


class TestGeneration(unittest.TestCase):
    def setUp(self):

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = GenerationSettings(api_key="unit-secret-key", max_retries=1)
        self.transport = Mock(side_effect=lambda *a, **kw: response())
        self.sleep = Mock()
        self.client = self.make_client()

    def make_client(self, settings=None):

        return NvidiaGenerationClient(settings or self.settings, self.root / "cache.sqlite3",
                                      self.root / "usage.jsonl", self.transport, self.sleep)

    def test_request_and_cache_reuse_without_key(self):

        self.assertEqual(self.client.generate_json("Return JSON", {"question": "private-text"}), {"ok": True})
        cached = self.make_client(replace(self.settings, api_key=""))
        self.assertEqual(cached.generate_json("Return JSON", {"question": "private-text"}), {"ok": True})
        self.assertEqual(cached.cache_hits, 1)
        self.assertEqual(self.transport.call_count, 1)
        request = self.transport.call_args.args[0]
        self.assertEqual(request.full_url, "https://integrate.api.nvidia.com/v1/chat/completions")
        body = json.loads(request.data)
        self.assertFalse(body["stream"])
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        log = (self.root / "usage.jsonl").read_text()
        for private in ("unit-secret-key", "private-text", "omit"):
            self.assertNotIn(private, log)
        self.assertEqual(json.loads(log)["usage"]["prompt_tokens"], 12)
        self.assertIsNone(json.loads(log)["estimated_cost_usd"])

    def test_prompt_model_and_token_limit_separate_cache(self):

        self.client.generate_json("v1", {})
        self.client.generate_json("v2", {})
        self.make_client(replace(self.settings, model="other/model")).generate_json("v1", {})
        self.make_client(replace(self.settings, max_tokens=1024)).generate_json("v1", {})
        self.assertEqual(self.transport.call_count, 4)

    def test_invalid_json_is_retried_once_and_then_cached(self):

        self.transport.side_effect = [response("not json"), response()]
        self.assertEqual(self.client.generate_json("JSON", {"retry": True}), {"ok": True})
        self.assertEqual(self.transport.call_count, 2)

    def test_missing_key_does_not_send_request(self):

        with self.assertRaises(GenerationError):
            self.make_client(replace(self.settings, api_key="")).generate_json("JSON", {})
        self.transport.assert_not_called()

    def test_damaged_cache_is_regenerated_and_closed(self):

        self.client.generate_json("JSON", {})
        connection = sqlite3.connect(self.root / "cache.sqlite3")
        try:
            connection.execute("UPDATE responses SET response = 'damaged'")
            connection.commit()
        finally:
            connection.close()
        self.client.generate_json("JSON", {})
        self.assertEqual(self.transport.call_count, 2)

    def test_authentication_error_not_retried_or_leaked(self):

        self.transport.side_effect = HTTPError("url", 401, "unit-secret-key", {}, io.BytesIO())
        with self.assertRaises(GenerationError) as caught:
            self.client.generate_json("JSON", {})
        self.assertNotIn("unit-secret-key", str(caught.exception))
        self.assertEqual(self.transport.call_count, 1)

    def test_404_identifies_model_without_retrying_or_leaking_provider_body(self):

        self.transport.side_effect = HTTPError("url", 404, "unit-secret-key", {}, io.BytesIO(b"unit-secret-key"))
        with self.assertRaises(GenerationError) as caught:
            self.client.generate_json("JSON", {})
        message = str(caught.exception)
        self.assertIn(self.settings.model, message)
        self.assertIn("--check-generation", message)
        self.assertNotIn("unit-secret-key", message)
        self.assertEqual(self.transport.call_count, 1)
        self.sleep.assert_not_called()

    def test_lightning_disables_thinking_and_keeps_embedding_configuration_separate(self):

        self.make_client(replace(self.settings, model="nvidia/nemotron-3.5-lightning-30b-a3b")).generate_json("JSON", {})
        body = json.loads(self.transport.call_args.args[0].data)
        self.assertEqual(body["model"], "nvidia/nemotron-3.5-lightning-30b-a3b")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_model_catalog_request_and_validation(self):

        self.transport.side_effect = [io.BytesIO(b'{"data":[{"id":"provider/b"},{"id":"provider/a"}]}'),
                                      io.BytesIO(b'{"data":[{}]}')]
        self.assertEqual(self.client.available_models(), ["provider/a", "provider/b"])
        request = self.transport.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, "https://integrate.api.nvidia.com/v1/models")
        with self.assertRaises(GenerationError):
            self.client.available_models()

    def test_model_catalog_authentication_failure_is_sanitized(self):

        self.transport.side_effect = HTTPError("url", 401, "unit-secret-key", {}, io.BytesIO())
        with self.assertRaises(GenerationError) as caught:
            self.client.available_models()
        self.assertNotIn("unit-secret-key", str(caught.exception))
        self.assertEqual(self.transport.call_count, 1)

    def test_transient_network_and_rate_limit_retry(self):

        for failure in (URLError("offline"), HTTPError("url", 429, "limit", {"Retry-After": "1"}, io.BytesIO())):
            with self.subTest(failure=type(failure).__name__):
                self.transport.side_effect = [failure, response()]
                self.client.generate_json("JSON", {"case": str(type(failure))})
        self.assertEqual(self.transport.call_count, 4)
        self.assertGreaterEqual(self.sleep.call_count, 2)

    def test_rate_limit_retries_are_observable_then_raise_429(self):

        events = []
        self.client.on_event = events.append
        self.transport.side_effect = lambda *a, **kw: (_ for _ in ()).throw(
            HTTPError("url", 429, "limit", {"Retry-After": "3"}, io.BytesIO()))
        with self.assertRaises(GenerationError) as caught:
            self.client.generate_json("JSON", {"case": "limited"})
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(caught.exception.retry_after, "3")
        self.assertEqual(self.transport.call_count, 2)
        self.assertEqual([event["stage"] for event in events], ["rate_limit_retry"])
        self.assertEqual(events[0]["details"]["wait_seconds"], 3)
        self.assertNotIn("unit-secret-key", json.dumps(events))

    def test_retry_limit(self):

        self.transport.side_effect = TimeoutError()
        with self.assertRaises(GenerationError):
            self.client.generate_json("JSON", {})
        self.assertEqual(self.transport.call_count, 2)

    def test_invalid_or_truncated_response_is_not_cached(self):

        for content, finish in (("[]", "stop"), ("```json\n{}\n```", "stop"), ('{"ok":true}', "length")):
            with self.subTest(content=content, finish=finish):
                self.transport.side_effect = lambda *a, **kw: response(content, finish)
                with self.assertRaises(GenerationError):
                    self.client.generate_json("JSON", {})
        self.transport.side_effect = lambda *a, **kw: response()
        self.assertEqual(self.client.generate_json("JSON", {}), {"ok": True})
        self.assertEqual(self.transport.call_count, 7)

    def test_settings_environment_and_validation(self):

        path = self.root / ".env"
        path.write_text("NVIDIA_API_KEY=from-file\nGENERATION_MAX_TOKENS=1024\n")
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "from-process"}, clear=True):
            settings = load_generation_settings(path)
        self.assertEqual(settings.api_key, "from-process")
        self.assertEqual(settings.max_tokens, 1024)
        self.assertNotIn("from-process", repr(settings))
        for values in ({"model": ""}, {"max_tokens": 0}, {"timeout": float("nan")},
                       {"max_retries": 9}, {"requests_per_minute": 0}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                GenerationSettings(**values)


if __name__ == "__main__":
    unittest.main()
