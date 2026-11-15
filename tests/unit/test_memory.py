"""Offline checks for immutable evidence-backed memory and temporal boundaries."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import unittest

from app.memory import (build_memory_prompt, memory_as_of, record_to_dict, source_turns,
                        validate_extraction, validate_ledger, validate_record)


def meeting(meeting_id):
    """Create a tiny transcript with exact character offsets and original wording."""

    texts = ["We agreed to use normal batteries only.",
             "Speaker B will check the battery prices tomorrow.",
             "No.", "We decided to use solar power instead."]
    raw = "\n".join(texts)
    turns = []
    position = 0
    for index, text in enumerate(texts, start=1):
        turns.append({"utterance_id": index, "speaker_id": "B" if index == 3 else "A",
                      "timestamp": f"00:00:0{index}", "original_text": text,
                      "start_char": position, "end_char": position + len(text)})
        position += len(text) + 1
    return {"meeting_id": meeting_id, "team_id": meeting_id[:-1],
            "meeting_sequence": meeting_id[-1], "meeting_number": "abcd".index(meeting_id[-1]) + 1,
            "original_transcript_text": raw, "utterances": turns}


class TestMemory(unittest.TestCase):
    def setUp(self):

        self.meetings = {identifier: meeting(identifier) for identifier in ("ES2002a", "ES2002c", "ES2002d", "TS3012c")}
        self.record = {
            "event_id": "ES2002a_e001", "type": "decision", "team_id": "ES2002", "meeting_id": "ES2002a",
            "speaker_id": "A", "topic": "power source", "summary": "Use normal batteries.",
            "source_turn_ids": [1], "evidence": [{"turn_id": 1, "quote": self.meetings["ES2002a"]["utterances"][0]["original_text"]}],
            "status": "agreed", "owner": None, "owner_source_turn_ids": [], "supersedes": [],
        }

    def later(self, meeting_id="ES2002c"):

        return dict(deepcopy(self.record), event_id=meeting_id + "_e002", meeting_id=meeting_id,
                    source_turn_ids=[4], evidence=[{"turn_id": 4, "quote": "We decided to use solar power instead."}],
                    summary="Use solar power.", supersedes=[self.record["event_id"]])

    def action(self):

        return dict(deepcopy(self.record), type="action_item", status="open", source_turn_ids=[2],
                    evidence=[{"turn_id": 2, "quote": "Speaker B will check the battery prices tomorrow."}],
                    owner="B", owner_source_turn_ids=[2], summary="Check battery prices.")

    def test_valid_record_is_immutable_and_round_trips(self):

        result = validate_record(self.record, self.meetings)
        self.assertEqual(result.meeting_number, 1)
        self.assertEqual(record_to_dict(result), self.record)
        with self.assertRaises(FrozenInstanceError):
            result.summary = "Changed"

    def test_reject_unknown_type_status_and_fields(self):

        for changes in ({"type": "guess"}, {"status": "open"}, {"summary": " "}, {"extra": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_record(dict(self.record, **changes), self.meetings)
        incomplete = dict(self.record)
        incomplete.pop("owner")
        with self.assertRaises(ValueError):
            validate_record(incomplete, self.meetings)

    def test_reject_bad_source_ids(self):

        for ids in ([], [True], [0], [999], [1, 1]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_record(dict(self.record, source_turn_ids=ids), self.meetings)

    def test_reject_wrong_team_or_cutoff(self):

        with self.assertRaisesRegex(ValueError, "team"):
            validate_record(dict(self.record, team_id="TS3012"), self.meetings)
        with self.assertRaisesRegex(ValueError, "cutoff"):
            validate_record(self.later("ES2002d"), self.meetings, as_of="c")

    def test_reject_wrong_speaker_and_invented_quotes(self):

        with self.assertRaisesRegex(ValueError, "speaker"):
            validate_record(dict(self.record, speaker_id="B"), self.meetings)
        for quote in ("normal batteries", "We agreed to use rechargeable batteries only."):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                validate_record(dict(self.record, evidence=[{"turn_id": 1, "quote": quote}]), self.meetings)

    def test_short_reply_can_support_but_not_supply_citation(self):

        record = dict(deepcopy(self.record), source_turn_ids=[1, 3])
        self.assertTrue(validate_record(record, self.meetings))
        record["evidence"] = [{"turn_id": 3, "quote": "No."}]
        with self.assertRaises(ValueError):
            validate_record(record, self.meetings)

    def test_reject_tampered_or_negative_offsets(self):

        for change in ({"original_text": "Changed source"}, {"start_char": -1}):
            meetings = deepcopy(self.meetings)
            meetings["ES2002a"]["utterances"][0].update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_record(self.record, meetings)

    def test_owner_can_differ_from_attributed_speaker(self):

        result = validate_record(self.action(), self.meetings)
        self.assertEqual(result.owner, "B")
        self.assertEqual(result.speaker_id, "A")

    def test_unknown_owner_is_explicitly_allowed(self):

        result = validate_record(dict(self.action(), owner=None, owner_source_turn_ids=[]), self.meetings)
        self.assertIsNone(result.owner)

    def test_owner_needs_known_speaker_and_assignment_evidence(self):

        for changes in ({"owner": "Z"}, {"owner_source_turn_ids": []},
                        {"owner_source_turn_ids": [1]}, {"owner": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_record(dict(self.action(), **changes), self.meetings)
        with self.assertRaises(ValueError):
            validate_record(dict(self.record, owner="A", owner_source_turn_ids=[1]), self.meetings)

    def test_completed_action_requires_evidence_but_is_not_semantically_judged(self):

        # Validation cannot decide if the text actually proves completion.
        self.assertEqual(validate_record(dict(self.action(), status="completed"), self.meetings).status, "completed")
        with self.assertRaises(ValueError):
            validate_record(dict(self.action(), status="completed", evidence=[]), self.meetings)

    def test_timeline_uses_explicit_links_without_mutating_inputs(self):

        records = [self.later(), self.record]
        snapshot = deepcopy(records)
        timeline = memory_as_of(records, self.meetings, "ES2002")
        self.assertEqual([item.status for item in timeline], ["superseded", "agreed"])
        self.assertEqual(records, snapshot)

    def test_future_link_does_not_change_earlier_snapshot(self):

        before = memory_as_of([self.record, self.later()], self.meetings, "ES2002", "b")
        self.assertEqual(len(before), 1)
        self.assertEqual(before[0].status, "agreed")

    def test_latest_meeting_alone_does_not_supersede(self):

        later = dict(self.later(), supersedes=[])
        timeline = memory_as_of([self.record, later], self.meetings, "ES2002")
        self.assertEqual([item.status for item in timeline], ["agreed", "agreed"])

    def test_reject_missing_backward_self_or_cross_topic_links(self):

        for later in (dict(self.later(), supersedes=["missing"]),
                      dict(self.later(), supersedes=["ES2002c_e002"]),
                      dict(self.later(), topic="case material"),
                      dict(self.later(), type="number", status="recorded"),
                      dict(self.later(), type="proposal", status="proposed")):
            with self.subTest(later=later["type"]), self.assertRaises(ValueError):
                validate_ledger([self.record, later], self.meetings)
        first = dict(self.record, supersedes=["ES2002c_e002"])
        with self.assertRaises(ValueError):
            validate_ledger([first, dict(self.later(), supersedes=[])], self.meetings)

    def test_reject_cross_team_supersession(self):

        later = dict(self.later("TS3012c"), team_id="TS3012")
        with self.assertRaisesRegex(ValueError, "same team"):
            validate_ledger([self.record, later], self.meetings)

    def test_same_meeting_links_follow_turn_order(self):

        later = dict(self.later("ES2002a"), event_id="ES2002a_e002")
        self.assertEqual(len(validate_ledger([self.record, later], self.meetings)), 2)

    def test_reject_duplicate_ids_and_unlinked_superseded_status(self):

        with self.assertRaises(ValueError):
            validate_ledger([self.record, self.record], self.meetings)
        with self.assertRaises(ValueError):
            validate_ledger([dict(self.record, status="superseded")], self.meetings)

    def test_prompt_contains_selected_turns_only_and_no_full_transcript(self):

        prompt = build_memory_prompt(self.meetings["ES2002a"], [2])
        context = json.loads(prompt["source_context"])
        self.assertEqual([item["turn_id"] for item in context["turns"]], [2])
        self.assertNotIn("normal batteries", prompt["source_context"])
        self.assertIn("untrusted", prompt["system"])

    def test_prompt_word_budget_rejects_instead_of_truncating(self):

        with self.assertRaises(ValueError):
            build_memory_prompt(self.meetings["ES2002a"], [1], max_words=2)

    def test_extraction_accepts_strict_json_and_empty_records(self):

        meeting = self.meetings["ES2002a"]
        records = validate_extraction(json.dumps({"records": [self.record]}), meeting, [1])
        self.assertEqual(len(records), 1)
        self.assertEqual(validate_extraction({"records": []}, meeting, [1]), [])

    def test_extraction_rejects_unsupplied_turns_and_prior_memory_links(self):

        meeting = self.meetings["ES2002a"]
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_extraction({"records": [self.record]}, meeting, [2])
        linked = dict(self.record, supersedes=["ES2002a_prior"])
        with self.assertRaisesRegex(ValueError, "reconcile"):
            validate_extraction({"records": [linked]}, meeting, [1])


if __name__ == "__main__":
    unittest.main()
