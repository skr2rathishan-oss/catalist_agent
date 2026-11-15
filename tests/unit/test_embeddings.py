"""Offline tests for NVIDIA requests, cache reuse, retry rules and validation."""

from dataclasses import replace
import io
import json
import math
import sqlite3
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from app.config import EmbeddingSettings, load_embedding_settings
from app.embeddings import normalise_vector, prepare_embedding_text
from app.llm import EmbeddingError, NvidiaEmbeddingClient


class TestNvidiaEmbeddings(unittest.TestCase):
    def setUp(self):

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.settings = EmbeddingSettings(api_key="fake-unit-test-key", max_retries=2)
        self.transport = Mock(side_effect=self.response)
        self.sleep = Mock()
        self.client = self.make_client()

    def make_client(self, settings=None):

        return NvidiaEmbeddingClient(
            settings or self.settings, self.root / "cache.sqlite3", self.root / "usage.jsonl",
            transport=self.transport, sleep=self.sleep,
        )

    def response(self, request, timeout):

        body = json.loads(request.data)
        embeddings = []
        for index, text in enumerate(body["input"]):
            values = [0.0] * body["dimensions"]
            values[0 if "battery" in text else 1] = 2.0
            embeddings.append({"index": index, "embedding": values})
        return io.BytesIO(json.dumps({"data": embeddings, "usage": {"prompt_tokens": 20}}).encode())

    def test_separate_documents_are_not_aggregated(self):

        vectors = self.client.embed(["battery", "solar"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual(vectors[0][0], 1)
        self.assertEqual(vectors[1][1], 1)
        request = self.transport.call_args.args[0]
        self.assertEqual(request.full_url, "https://integrate.api.nvidia.com/v1/embeddings")
        body = json.loads(request.data)
        self.assertEqual(body["input"], ["battery", "solar"])
        self.assertEqual(body["model"], "nvidia/nemotron-3.5-embed-8b")
        self.assertEqual(body["input_type"], "passage")
        self.assertEqual(body["dimensions"], 4096)
        self.assertEqual(body["truncate"], "NONE")
        self.assertEqual(request.get_header("Authorization"), "Bearer fake-unit-test-key")

    def test_cache_reuse_and_duplicate_order(self):

        first = self.client.embed(["battery", "solar", "battery"])
        second = self.make_client().embed(["solar", "battery"])
        self.assertEqual(self.transport.call_count, 1)
        self.assertEqual(first[0], first[2])
        self.assertEqual(second, [first[1], first[0]])

    def test_cache_separates_query_and_document(self):

        self.client.embed(["battery"])
        self.client.embed(["battery"], task="query")
        self.assertEqual(self.transport.call_count, 2)
        request = self.transport.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body["input"], ["battery"])
        self.assertEqual(body["input_type"], "query")

    def test_dimensions_change_invalidates_cache(self):

        self.client.embed(["battery"])
        larger = self.make_client(replace(self.settings, dimensions=1536))
        self.assertEqual(len(larger.embed(["battery"])[0]), 1536)
        self.assertEqual(self.transport.call_count, 2)

    def test_completed_batches_survive_later_failure(self):

        client = self.make_client(replace(self.settings, batch_size=1, max_retries=0))
        good = self.response
        attempts = []

        def fail_second(request, timeout):

            attempts.append(request)
            if len(attempts) == 2:
                raise URLError("offline")
            return good(request, timeout)

        self.transport.side_effect = fail_second
        with self.assertRaises(EmbeddingError):
            client.embed(["battery", "solar"])
        self.transport.side_effect = self.response
        self.transport.reset_mock()
        vectors = self.make_client().embed(["battery", "solar"])
        self.assertEqual(len(vectors), 2)
        pending = json.loads(self.transport.call_args.args[0].data)["input"]
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0], "solar")

    def test_timeouts_and_rate_limits_are_retried(self):

        request_error = HTTPError("https://example.invalid", 429, "rate limited", {"Retry-After": "1"}, None)
        self.transport.side_effect = [TimeoutError(), request_error, self.response_for_battery()]
        self.assertTrue(self.client.embed(["battery"]))
        self.assertEqual(self.transport.call_count, 3)
        self.assertTrue(self.sleep.called)

    def response_for_battery(self):

        values = [1.0] + [0.0] * (self.settings.dimensions - 1)
        return io.BytesIO(json.dumps({"data": [{"index": 0, "embedding": values}]}).encode())

    def test_authentication_errors_are_not_retried_or_leaked(self):

        self.transport.side_effect = HTTPError("https://example.invalid", 403, "fake-unit-test-key", {}, None)
        with self.assertRaises(EmbeddingError) as error:
            self.client.embed(["battery"])
        self.assertNotIn(self.settings.api_key, str(error.exception))
        self.assertEqual(self.transport.call_count, 1)

    def test_bad_response_is_rejected(self):

        self.transport.side_effect = None
        self.transport.return_value = io.BytesIO(b'{"data": []}')
        with self.assertRaisesRegex(EmbeddingError, "invalid"):
            self.client.embed(["battery"])

    def test_missing_key_allows_cached_reads_only(self):

        first = self.client.embed(["battery"])
        offline = self.make_client(replace(self.settings, api_key=""))
        self.assertEqual(offline.embed(["battery"]), first)
        with self.assertRaisesRegex(EmbeddingError, "NVIDIA_API_KEY"):
            offline.embed(["new passage"])

    def test_logs_have_usage_but_no_text_or_key(self):

        self.client.embed(["battery private passage"])
        content = (self.root / "usage.jsonl").read_text(encoding="utf-8")
        self.assertNotIn(self.settings.api_key, content)
        self.assertNotIn("private passage", content)
        self.assertEqual(json.loads(content)["usage"]["prompt_tokens"], 20)
        self.assertNotIn(self.settings.api_key, repr(self.settings))

    def test_empty_input_makes_no_request(self):

        self.assertEqual(self.client.embed([]), [])
        with self.assertRaises(ValueError):
            self.client.embed([" "])
        self.transport.assert_not_called()

    def test_invalid_vectors(self):

        for values in ([0, 0], [math.nan, 1], [math.inf, 1], [True, 1], ["1", 2], [1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                normalise_vector(values, 2)
        self.assertEqual(normalise_vector([3, 4], 2), [0.6, 0.8])

    def test_network_retries_are_bounded(self):

        self.transport.side_effect = URLError("offline")
        with self.assertRaises(EmbeddingError):
            self.client.embed(["battery"])
        self.assertEqual(self.transport.call_count, self.settings.max_retries + 1)

    def test_damaged_cache_entry_is_regenerated(self):

        expected = self.client.embed(["battery"])
        connection = sqlite3.connect(self.root / "cache.sqlite3")
        try:
            connection.execute("UPDATE embeddings SET vector = '[]'")
            connection.commit()
        finally:
            # SQLite's connection context manager does not close the file.
            # Release it explicitly so Windows can clean up this directory.
            connection.close()
        self.assertEqual(self.client.embed(["battery"]), expected)
        self.assertEqual(self.transport.call_count, 2)

    def test_invalid_settings_are_rejected(self):

        for settings in ({"dimensions": 0}, {"dimensions": 4097}, {"batch_size": 0}, {"requests_per_minute": 0},
                         {"timeout": math.nan}, {"max_retries": 6}, {"model": "unknown"}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                EmbeddingSettings(**settings)

    def test_task_formatting_preserves_language(self):

        text = "Why did the team reject kinetic power?"
        self.assertEqual(prepare_embedding_text(text, "query"), text)
        self.assertEqual(prepare_embedding_text(text, "document"), text)
        with self.assertRaises(ValueError):
            prepare_embedding_text(text, "unsupported")

    def test_environment_settings(self):

        env_file = self.root / ".env"
        env_file.write_text("NVIDIA_API_KEY=local-test-key\nEMBEDDING_DIMENSIONS=1536\n", encoding="utf-8")
        with patch.dict("os.environ", {"EMBEDDING_DIMENSIONS": "768"}, clear=True):
            settings = load_embedding_settings(env_file)
        self.assertEqual(settings.dimensions, 768)
        self.assertEqual(settings.api_key, "local-test-key")

    def test_out_of_order_responses_are_reassembled(self):

        def reversed_response(request, timeout):

            with self.response(request, timeout) as response:
                data = json.load(response)
            data["data"].reverse()
            return io.BytesIO(json.dumps(data).encode())

        self.transport.side_effect = reversed_response
        vectors = self.client.embed(["battery", "solar"])
        self.assertEqual(vectors[0][0], 1.0)
        self.assertEqual(vectors[1][1], 1.0)

    def test_duplicate_response_indices_are_rejected(self):

        values = [1.0] + [0.0] * (self.settings.dimensions - 1)
        data = {"data": [{"index": 0, "embedding": values}, {"index": 0, "embedding": values}]}
        self.transport.side_effect = None
        self.transport.return_value = io.BytesIO(json.dumps(data).encode())
        with self.assertRaisesRegex(EmbeddingError, "invalid"):
            self.client.embed(["battery", "solar"])


if __name__ == "__main__":
    unittest.main()
