"""Offline provider isolation, catalog filtering, fresh probes and rate-limit routing."""

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.config import GenerationSettings, load_generation_settings
from app.generation import ChatGenerationClient, GenerationError
from app.providers import discover_models, probe_model, usage_summary, GenerationRouter


class TestProviders(unittest.TestCase):
    def test_nvidia_settings_load_key_and_model_and_reject_other_providers(self):

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("NVIDIA_API_KEY=nvidia-secret\nGENERATION_MODEL=nvidia/model\n")
            with patch.dict("os.environ", {}, clear=True):
                settings = load_generation_settings(path)
                self.assertEqual((settings.provider, settings.api_key, settings.model),
                                 ("nvidia", "nvidia-secret", "nvidia/model"))
                self.assertEqual(load_generation_settings(path, model="nvidia/other").model, "nvidia/other")
                self.assertNotIn("secret", repr(settings))
                for provider in ("groq", "openrouter"):
                    with self.subTest(provider=provider), self.assertRaises(ValueError):
                        load_generation_settings(path, provider=provider)
            with self.assertRaises(ValueError):
                GenerationSettings(api_key="k", model="m", provider="groq")

    def test_requests_use_nvidia_endpoint_key_and_user_agent(self):

        def response(*args, **kwargs):
            data = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"ok": True})}}]}
            return io.BytesIO(json.dumps(data).encode())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transport = Mock(side_effect=response)
            config = GenerationSettings(api_key="nvidia-secret", model="same/model")
            client = ChatGenerationClient(config, root / "cache.db", root / "usage.jsonl", transport, Mock())
            self.assertEqual(client.generate_json("JSON", {}), {"ok": True})
            request = transport.call_args.args[0]
            self.assertEqual(request.full_url, "https://integrate.api.nvidia.com/v1/chat/completions")
            self.assertEqual(request.get_header("Authorization"), "Bearer nvidia-secret")
            # Some API fronts reject urllib's default agent (HTTP 403).
            self.assertEqual(request.get_header("User-agent"), "meeting-rag/1.0")
            self.assertNotIn("chat_template_kwargs", json.loads(request.data))
            self.assertNotIn("-secret", (root / "usage.jsonl").read_text())

    def test_catalog_filters_nontext_short_context_and_inactive(self):

        client = Mock()
        text = {"id": "nvidia/chat", "context_length": 32768}
        client.model_catalog.return_value = [text, dict(text, id="nvidia/whisper"),
                                            dict(text, id="nvidia/short", context_length=8192),
                                            dict(text, id="nvidia/off", active=False),
                                            dict(text, id="nvidia/unknown-context", context_length=None),
                                            dict(text, id="nvidia/picture", architecture={"output_modalities": ["image"]})]
        rows = discover_models(client=client)
        self.assertEqual([row["model"] for row in rows], ["nvidia/chat", "nvidia/unknown-context"])
        self.assertEqual(rows[0]["provider"], "nvidia")

    def test_probes_are_fresh_and_do_not_claim_rag_accuracy(self):

        client = Mock()
        client.generate_json.return_value = {"ok": True}
        self.assertEqual(probe_model("nvidia", "model", client)["status"], "JSON check passed")
        probe_model("nvidia", "model", client)
        calls = client.generate_json.call_args_list
        self.assertNotEqual(calls[0].args[1]["nonce"], calls[1].args[1]["nonce"])
        client.generate_json.side_effect = GenerationError("limited", status=429)
        self.assertEqual(probe_model("nvidia", "model", client)["status"], "rate limited")

    def test_router_switches_on_429_and_stays_on_backup(self):

        first, second = Mock(), Mock()
        first.settings = SimpleNamespace(provider="nvidia", model="first")
        second.settings = SimpleNamespace(provider="nvidia", model="second")
        first.generate_json.side_effect = GenerationError("limited", status=429)
        second.generate_json.return_value = {"ok": True}
        first.api_requests, second.api_requests = 1, 2
        first.cache_hits, second.cache_hits = 0, 1
        events = []
        factory = Mock(side_effect=[first, second])
        router = GenerationRouter([("nvidia", "first"), ("nvidia", "second")], events.append, factory)
        router.generate_json("JSON", {})
        router.generate_json("JSON", {})
        self.assertEqual(first.generate_json.call_count, 1)
        self.assertEqual(second.generate_json.call_count, 2)
        self.assertEqual(router.settings.model, "second")
        self.assertEqual(events[0]["stage"], "model_fallback")
        self.assertEqual(router.api_requests, 3)
        self.assertEqual(router.cache_hits, 1)

    def test_router_switches_after_exhausted_network_failure(self):

        first, second = Mock(), Mock()
        first.settings = SimpleNamespace(provider="nvidia", model="first")
        second.settings = SimpleNamespace(provider="nvidia", model="second")
        first.generate_json.side_effect = GenerationError("timed out", status="network_error")
        second.generate_json.return_value = {"ok": True}
        events = []
        router = GenerationRouter([("nvidia", "first"), ("nvidia", "second")], events.append,
                                  Mock(side_effect=[first, second]))
        self.assertEqual(router.generate_json("JSON", {}), {"ok": True})
        self.assertEqual(router.settings.model, "second")
        self.assertEqual(events[0]["details"]["reason"], "network error")

    def test_router_does_not_switch_for_authentication_or_bad_json(self):

        for status in (401, 403, 404, "invalid_response", None):
            with self.subTest(status=status):
                first, second = Mock(), Mock()
                first.generate_json.side_effect = GenerationError("failed", status=status)
                router = GenerationRouter([("nvidia", "first"), ("nvidia", "second")], client_factory=Mock(side_effect=[first, second]))
                with self.assertRaises(GenerationError):
                    router.generate_json("JSON", {})
                second.generate_json.assert_not_called()

    def test_router_stops_when_selected_routes_are_rate_limited(self):

        clients = [Mock(), Mock()]
        for client, name in zip(clients, ("first", "second")):
            client.settings = SimpleNamespace(provider="nvidia", model=name)
            client.generate_json.side_effect = GenerationError("limited", status=429)
        router = GenerationRouter([("nvidia", "first"), ("nvidia", "second")], client_factory=Mock(side_effect=clients))
        with self.assertRaises(GenerationError):
            router.generate_json("JSON", {})
        self.assertTrue(all(client.generate_json.call_count == 1 for client in clients))

    def test_usage_summary_flags_rate_limited_models(self):

        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as directory:
            generation, embedding = Path(directory) / "gen.jsonl", Path(directory) / "emb.jsonl"
            rows = [("2026-01-01T09:00:00+00:00", "nvidia", "m1", "ok"),
                    ("2026-01-01T09:10:00+00:00", "nvidia", "m1", 429),
                    ("2026-01-01T09:20:00+00:00", "nvidia", "m2", "ok"),
                    ("2025-12-31T09:00:00+00:00", "nvidia", "old", 429)]
            generation.write_text("\n".join(json.dumps({"time": t, "provider": p, "model": m, "status": s})
                                            for t, p, m, s in rows) + "\nnot json\n")
            embedding.write_text(json.dumps({"time": "2026-01-01T09:15:00+00:00", "model": "emb", "status": 429}) + "\n")
            now = datetime(2026, 1, 1, 9, 30, tzinfo=timezone.utc)
            summary = {row["model"]: row for row in usage_summary(generation, embedding, now=now)}
            self.assertNotIn("old", summary)
            self.assertEqual(summary["m1"]["state"], "rate limited")
            self.assertEqual((summary["m1"]["requests"], summary["m1"]["ok"], summary["m1"]["rate_limited"]), (2, 1, 1))
            self.assertEqual(summary["m2"]["state"], "ok")
            self.assertEqual(summary["emb"]["kind"], "embedding")
            self.assertEqual(usage_summary(Path(directory) / "missing", Path(directory) / "missing2"), [])

    def test_router_rejects_excessive_or_duplicate_routes(self):

        for routes in ([], [("nvidia", "m")] * 2, [("nvidia", str(i)) for i in range(4)]):
            with self.subTest(routes=routes), self.assertRaises(ValueError):
                GenerationRouter(routes)


if __name__ == "__main__":
    unittest.main()
