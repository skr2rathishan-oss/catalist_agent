"""Offline extraction, idempotence, replacement, and cutoff checks."""

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from app.generation import GenerationError
from app.memory_pipeline import extract_meeting, load_team_memory, memory_hints, reconcile_team, turn_windows


def meeting(meeting_id, text):
    return {"meeting_id": meeting_id, "team_id": meeting_id[:-1],
            "meeting_sequence": meeting_id[-1],
            "meeting_number": "abcd".index(meeting_id[-1]) + 1,
            "original_transcript_text": text,
            "utterances": [{"utterance_id": 1, "speaker_id": "A",
                            "timestamp": "00:00:01", "original_text": text,
                            "start_char": 0, "end_char": len(text)}]}


class FakeClient:
    def __init__(self):
        self.settings = SimpleNamespace(model="fake-model")
        self.calls = 0

    def generate_json(self, system, payload):
        self.calls += 1
        if "earlier_events" in payload:
            return {"supersedes": [payload["earlier_events"][0]["event_id"]]}
        context = payload["source_context"]
        text = context["turns"][0]["original_text"]
        return {"records": [{
            "event_id": context["meeting_id"] + "_draft",
            "type": "decision", "team_id": context["team_id"],
            "meeting_id": context["meeting_id"], "speaker_id": "A",
            "topic": "power source", "summary": text,
            "source_turn_ids": [1], "evidence": [{"turn_id": 1, "quote": text}],
            "status": "agreed", "owner": None, "owner_source_turn_ids": [],
            "supersedes": [],
        }]}


class TestMemoryPipeline(unittest.TestCase):
    def test_extract_reconcile_and_cutoff(self):
        first = meeting("ES2002a", "We agreed to use normal batteries only.")
        later = meeting("ES2002c", "We decided to use solar power instead.")
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed, memory = root / "parsed", root / "memory"
            parsed.mkdir()
            for item in (first, later):
                (parsed / f"{item['meeting_id']}.json").write_text(
                    json.dumps(item), encoding="utf-8")
                self.assertFalse(extract_meeting(item, client, memory)["cached"])
            self.assertEqual(client.calls, 2)
            self.assertTrue(extract_meeting(first, client, memory)["cached"])
            self.assertEqual(client.calls, 2)
            result = reconcile_team("ES2002", client, parsed, memory)
            self.assertEqual(result["links"], 1)
            self.assertTrue(reconcile_team("ES2002", client, parsed, memory)["cached"])
            early = load_team_memory("ES2002", "a", parsed, memory)
            current = load_team_memory("ES2002", "c", parsed, memory)
            self.assertEqual([item.status for item in early], ["agreed"])
            self.assertEqual([item.status for item in current], ["superseded", "agreed"])
            hints = memory_hints("Which power source?", "ES2002", "c", parsed, memory)
            self.assertTrue(hints)
            self.assertEqual(hints[0]["meeting_id"], "ES2002c")

    def test_window_budget_rejects_oversized_turn(self):
        item = meeting("ES2002a", "word " * 7)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            turn_windows(item, max_words=6)

    def test_empty_meeting_memory_is_cached(self):
        item = meeting("ES2002a", "We talked without recording any decision or action.")
        class EmptyClient(FakeClient):
            def generate_json(self, system, payload):
                self.calls += 1
                return {"records": []}
        client = EmptyClient()
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(extract_meeting(item, client, Path(directory))["records"], 0)
            self.assertTrue(extract_meeting(item, client, Path(directory))["cached"])
            self.assertEqual(client.calls, 1)

    def test_unambiguous_missing_contract_fields_are_filled(self):
        item = meeting("ES2002a", "We agreed to use normal batteries only.")
        class OmittedFieldsClient(FakeClient):
            def generate_json(self, system, payload):
                result = super().generate_json(system, payload)
                for field in ("status", "owner", "owner_source_turn_ids", "supersedes"):
                    result["records"][0].pop(field)
                return result
        client = OmittedFieldsClient()
        with tempfile.TemporaryDirectory() as directory:
            result = extract_meeting(item, client, Path(directory))
            self.assertEqual(result["records"], 1)
            self.assertEqual(result["rejected_records"], 0)
            self.assertEqual(client.calls, 1)

    def test_retry_preserves_better_saved_memory(self):
        item = meeting("ES2002a", "We agreed to use normal batteries only.")
        class MixedClient(FakeClient):
            def generate_json(self, system, payload):
                if "invalid_records" in payload:
                    self.calls += 1
                    return {"records": []}
                result = super().generate_json(system, payload)
                result["records"].append(dict(result["records"][0], type="unknown"))
                return result
        class EmptyClient(FakeClient):
            def generate_json(self, system, payload):
                self.calls += 1
                return {"records": []}
        with tempfile.TemporaryDirectory() as directory:
            memory = Path(directory)
            first = extract_meeting(item, MixedClient(), memory)
            self.assertEqual((first["records"], first["rejected_records"]), (1, 1))
            saved = (memory / "ES2002a.json").read_bytes()
            second = extract_meeting(item, EmptyClient(), memory, retry_rejected=True)
            self.assertTrue(second["retained_existing"])
            self.assertEqual((memory / "ES2002a.json").read_bytes(), saved)

    def test_reconciliation_checks_related_topic_wording(self):
        first = meeting("ES2002a", "We decided on normal batteries for power.")
        later = meeting("ES2002c", "We changed the battery choice to rechargeable cells.")
        class DifferentTopicsClient(FakeClient):
            def generate_json(self, system, payload):
                result = super().generate_json(system, payload)
                if "records" in result:
                    result["records"][0]["topic"] = (
                        "battery power" if payload["source_context"]["meeting_id"].endswith("a")
                        else "battery choice")
                return result
        client = DifferentTopicsClient()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed, memory = root / "parsed", root / "memory"
            parsed.mkdir()
            for item in (first, later):
                (parsed / f"{item['meeting_id']}.json").write_text(json.dumps(item), encoding="utf-8")
                extract_meeting(item, client, memory)
            self.assertEqual(reconcile_team("ES2002", client, parsed, memory)["links"], 1)

    def test_fallback_recovers_window_without_source_valid_primary_records(self):
        item = meeting("ES2002a", "We agreed to use normal batteries only.")
        class WrongTeamClient(FakeClient):
            def generate_json(self, system, payload):
                result = super().generate_json(system, payload)
                result["records"][0]["team_id"] = "TS3012"
                return result
        with tempfile.TemporaryDirectory() as directory:
            result = extract_meeting(item, WrongTeamClient(), Path(directory),
                                     fallback_client=FakeClient())
            self.assertEqual(result["records"], 1)
            self.assertEqual(result["fallback_windows"], 1)
            self.assertEqual(result["rejected_records"], 0)

    def test_invalid_long_window_splits_on_source_turns(self):
        first = "We agreed to use normal batteries only."
        second = "We agreed to use a bright display too."
        source = first + "\n" + second
        item = meeting("ES2002a", source)
        item["utterances"] = [dict(item["utterances"][0], original_text=first,
                                   start_char=0, end_char=len(first)),
                              dict(item["utterances"][0], utterance_id=2,
                                   original_text=second, start_char=len(first) + 1,
                                   end_char=len(source))]
        class FailsLongClient(FakeClient):
            def generate_json(self, system, payload):
                if len(payload["source_context"]["turns"]) > 1:
                    raise GenerationError("Invalid JSON", status="invalid_response")
                result = super().generate_json(system, payload)
                turn_id = payload["source_context"]["turns"][0]["turn_id"]
                result["records"][0]["source_turn_ids"] = [turn_id]
                result["records"][0]["evidence"][0]["turn_id"] = turn_id
                return result
        with tempfile.TemporaryDirectory() as directory:
            result = extract_meeting(item, FailsLongClient(), Path(directory))
            self.assertEqual(result["records"], 2)
            self.assertEqual(result["windows"], 2)

    def test_empty_long_response_is_recovered_by_smaller_windows(self):
        first = "We agreed on normal batteries. " + "Discussion continued. " * 250
        second = "We agreed on a bright display. " + "Discussion continued. " * 250
        source = first + "\n" + second
        item = meeting("ES2002a", source)
        item["utterances"] = [dict(item["utterances"][0], original_text=first,
                                   start_char=0, end_char=len(first)),
                              dict(item["utterances"][0], utterance_id=2,
                                   original_text=second, start_char=len(first) + 1,
                                   end_char=len(source))]
        class EmptyLongClient(FakeClient):
            def generate_json(self, system, payload):
                turns = payload["source_context"]["turns"]
                if len(turns) > 1:
                    return {"records": []}
                result = super().generate_json(system, payload)
                result["records"][0]["source_turn_ids"] = [turns[0]["turn_id"]]
                result["records"][0]["evidence"][0]["turn_id"] = turns[0]["turn_id"]
                return result
        with tempfile.TemporaryDirectory() as directory:
            result = extract_meeting(item, EmptyLongClient(), Path(directory))
            self.assertEqual(result["records"], 2)
            self.assertEqual(result["windows"], 2)


if __name__ == "__main__":
    unittest.main()
