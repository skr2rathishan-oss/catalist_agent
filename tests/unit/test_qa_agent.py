"""Offline tests of question planning, gap searches, and grounded answer checks."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from app.agents.qa_agent import QuestionAnsweringAgent, citation_options
from app.agents.prompts import QA_COVERAGE


class TestQuestionAnsweringAgent(unittest.TestCase):
    def setUp(self):

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.first = "We agreed the remote will use replaceable batteries."
        self.second = "The marketing expert will check the prices tomorrow."
        text = self.first + "\n" + self.second
        turns = []
        for number, (quote, start) in enumerate(((self.first, 0), (self.second, len(self.first) + 1)), 1):
            turns.append({"utterance_id": number, "speaker_id": "B", "timestamp": "00:01:00",
                          "original_text": quote, "start_char": start, "end_char": start + len(quote)})
        self.meeting = {"meeting_id": "ES2002b", "team_id": "ES2002", "meeting_sequence": "b",
                        "meeting_number": 2, "original_transcript_text": text, "utterances": turns}
        (self.root / "ES2002b.json").write_text(json.dumps(self.meeting), encoding="utf-8")
        self.chunk = {"chunk_id": "ES2002b_c001", "meeting_id": "ES2002b", "team_id": "ES2002",
                      "turn_ids": [1]}
        self.search = Mock(return_value=[self.chunk])
        self.client = Mock()
        self.plan = {"parts": [{"question": "Which power source?", "query": "final power source",
                                "requirements": ["What source was chosen?"]}]}
        self.coverage = {"parts": [self.support("p1", 1)]}
        self.draft = {"answer": "The team chose replaceable batteries.",
                      "citations": [dict(self.evidence(1), part_ids=["p1"])]}
        self.review = {"approved": True, "issues": []}

    def evidence(self, turn):

        return {"meeting_id": "ES2002b", "turn_id": turn, "quote": self.first if turn == 1 else self.second}

    def support(self, part, turn):

        return {"id": part, "requirements": [{"id": f"{part}_r1", "evidence": [self.evidence(turn)]}], "next_query": None}

    def missing(self, part, query="marketing price follow up"):

        return {"id": part, "requirements": [{"id": f"{part}_r1", "evidence": []}], "next_query": query}

    def agent(self, responses, **kwargs):

        self.client.generate_json.side_effect = deepcopy(responses)
        return QuestionAnsweringAgent(self.client, self.search, self.root, **kwargs)

    def test_success_and_submission_citation_shape(self):

        result = self.agent([self.plan, self.coverage, self.draft, self.review]).answer("Which power source?", "es2002", "c")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"], [{"meeting_id": "ES2002b", "quote": self.first}])
        self.search.assert_called_once_with("final power source", "ES2002", as_of="c", limit=5)
        coverage_prompt = self.client.generate_json.call_args_list[1]
        self.assertEqual(coverage_prompt.args[0], QA_COVERAGE)
        self.assertEqual(len(coverage_prompt.args[1]["sources"]), 1)

    def test_trailing_schema_field_name_is_removed_from_answer_prose(self):

        draft = deepcopy(self.draft)
        draft["answer"] += " citations"
        result = self.agent([self.plan, self.coverage, draft, self.review]).answer("Which power source?", "ES2002")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], self.draft["answer"])

    def test_requirement_part_ids_and_inline_markers_are_canonicalized(self):

        draft = deepcopy(self.draft)
        draft["answer"] += (' [ES2002b_t1_w0] �ES2002b_t1_w0 '
                            '[citations:[{"citation_id":"ES2002b_t1_w0","part_ids":["p1"]}]] '
                            'Citation[{"citation_id":"","part_ids":["p1"]}]')
        draft["citations"][0]["part_ids"] = ["p1_r1"]
        result = self.agent([self.plan, self.coverage, draft, self.review]).answer("Which power source?", "ES2002")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["answer"], self.draft["answer"])

    def test_original_question_phase_filters_even_a_bad_planner_query(self):

        self.plan["parts"][0]["query"] = "power source meeting d"
        result = self.agent([self.plan, self.coverage, self.draft, self.review]).answer(
            "What power source was discussed in the ES2002 functional design meeting?", "ALL")
        self.assertEqual(result["status"], "answered")
        self.search.assert_called_once_with("power source meeting d", "ES2002", as_of="d", limit=5, meeting_number=2)
        payload = self.client.generate_json.call_args_list[0].args[1]
        self.assertEqual(payload["target_meeting"], "b")
        self.assertEqual(payload["as_of"], "d")

    def test_planner_exclusion_instruction_is_not_an_evidence_requirement(self):

        plan = deepcopy(self.plan)
        plan["parts"][0]["requirements"].extend([
            "Unresolved proposals must be excluded", "ensure excluded are speculative ideas"])
        result = self.agent([plan, self.coverage, self.draft, self.review]).answer(
            "Which power source was chosen? Exclude unresolved proposals.", "ES2002")
        self.assertEqual(result["status"], "answered")
        coverage_payload = self.client.generate_json.call_args_list[1].args[1]
        self.assertEqual([item["question"] for item in coverage_payload["parts"][0]["requirements"]],
                         ["What source was chosen?"])

    def test_wrong_phase_and_future_requested_phase_are_rejected(self):

        with self.assertRaisesRegex(ValueError, "requested meeting"):
            self.agent([self.plan]).answer("What happened in ES2002a?", "ES2002")
        self.client.reset_mock()
        with self.assertRaisesRegex(ValueError, "cutoff"):
            self.agent([]).answer("What happened in ES2002c?", "ES2002", "b")
        self.client.generate_json.assert_not_called()

    def requirement_plan(self):

        return {"parts": [{"question": "Power source and price owner?", "query": "power source price owner",
                           "requirements": ["What power source?", "Who checks prices?"]}]}

    def requirement_coverage(self, complete):

        return {"parts": [{"id": "p1", "requirements": [
            {"id": "p1_r1", "evidence": [self.evidence(1)]},
            {"id": "p1_r2", "evidence": [self.evidence(2)] if complete else []}],
            "next_query": None if complete else "price check assignment owner"}]}

    def test_missing_atomic_fact_requires_followup_before_drafting(self):

        self.search.side_effect = [[self.chunk], [dict(self.chunk, turn_ids=[2])]]
        draft = dict(self.draft, answer="The team chose batteries; the marketing expert will check prices.",
                     citations=[dict(self.evidence(1), part_ids=["p1"]), dict(self.evidence(2), part_ids=["p1"])])
        result = self.agent([self.requirement_plan(), self.requirement_coverage(False),
                             self.requirement_coverage(True), draft, self.review]).answer("Power and price owner?", "ES2002")
        self.assertEqual(result["status"], "answered")
        self.assertEqual([call.args[0] for call in self.search.call_args_list],
                         ["power source price owner", "price check assignment owner"])
        self.assertEqual(result["trace"][1]["missing_parts"], ["p1"])
        self.assertFalse(result["trace"][1]["requirement_checks"][1]["supported"])

    def test_supported_coverage_ignores_redundant_followup_query(self):

        coverage = self.requirement_coverage(True)
        coverage["parts"][0]["next_query"] = "unnecessary additional search"
        self.search.return_value = [dict(self.chunk, turn_ids=[1, 2])]
        agent = self.agent([])
        meetings, sources = {}, {}
        agent.add_sources(self.search.return_value, meetings, sources, "ES2002", "d")
        planned = dict(self.requirement_plan()["parts"][0], id="p1", requirements=[
            {"id": "p1_r1", "question": "Power?"}, {"id": "p1_r2", "question": "Owner?"}])
        checked, pending = agent.check_coverage(coverage, [planned], meetings, sources, "ES2002", "d")
        self.assertFalse(pending)
        self.assertIsNone(checked["parts"][0]["next_query"])

    def test_generic_part_boolean_cannot_skip_atomic_evidence_checks(self):

        unchecked = {"parts": [{"id": "p1", "supported": True, "evidence": [self.evidence(1)], "next_query": None}]}
        result = self.agent([self.requirement_plan(), unchecked, unchecked]).answer("Power and price owner?", "ES2002")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertEqual(self.client.generate_json.call_count, 3)

    def test_final_citations_cannot_omit_a_checked_requirement(self):

        self.search.return_value = [dict(self.chunk, turn_ids=[1, 2])]
        result = self.agent([self.requirement_plan(), self.requirement_coverage(True),
                             self.draft, self.draft, self.draft]).answer("Power and price owner?", "ES2002")
        self.assertEqual(result["status"], "invalid_answer")
        self.assertIn("p1_r2", result["trace"][-1]["validation_error"])

    def test_requirement_coverage_rejects_missing_duplicate_and_unknown_ids(self):

        agent = self.agent([])
        meetings, sources = {}, {}
        agent.add_sources([dict(self.chunk, turn_ids=[1, 2])], meetings, sources, "ES2002", "d")
        planned = dict(self.requirement_plan()["parts"][0], id="p1", requirements=[
            {"id": "p1_r1", "question": "Power?"}, {"id": "p1_r2", "question": "Owner?"}])
        for identifiers in (("p1_r1",), ("p1_r1", "p1_r1"), ("p1_r1", "p1_r3")):
            response = {"parts": [{"id": "p1", "requirements": [
                {"id": identifier, "evidence": [self.evidence(1)]} for identifier in identifiers], "next_query": None}]}
            with self.subTest(identifiers=identifiers), self.assertRaises(ValueError):
                agent.check_coverage(response, [planned], meetings, sources, "ES2002", "d")

    def test_budget_is_shared_across_passages_instead_of_first_chunk_only(self):

        texts = ["first " * 500, "second " * 400, "third " * 300]
        text, turns = "", []
        for number, original in enumerate(texts, 1):
            start = len(text)
            text += original + "\n"
            turns.append({"utterance_id": number, "speaker_id": "B", "timestamp": "00:01:00",
                          "original_text": original, "start_char": start, "end_char": start + len(original)})
        meeting = dict(self.meeting, original_transcript_text=text, utterances=turns)
        (self.root / "ES2002b.json").write_text(json.dumps(meeting), encoding="utf-8")
        sources, meetings, diagnostics = {}, {}, {}
        self.agent([]).add_sources([dict(self.chunk, turn_ids=[1, 2]),
                                   dict(self.chunk, chunk_id="ES2002b_c002", turn_ids=[3])],
                                  meetings, sources, "ES2002", "d", diagnostics=diagnostics)
        self.assertEqual(set(sources), {("ES2002b", 1), ("ES2002b", 3)})
        self.assertEqual([item["turn_id"] for item in diagnostics["dropped"]], [2])

    def test_query_matching_turn_is_admitted_before_earlier_chunk_context(self):

        texts = ["background " * 330, "more context " * 330,
                 "The selected design has a curvy fruit texture and colour."]
        text, turns = "", []
        for number, original in enumerate(texts, 1):
            start = len(text)
            text += original + "\n"
            turns.append({"utterance_id": number, "speaker_id": "B", "timestamp": "00:01:00",
                          "original_text": original, "start_char": start, "end_char": start + len(original)})
        meeting = dict(self.meeting, original_transcript_text=text, utterances=turns)
        (self.root / "ES2002b.json").write_text(json.dumps(meeting), encoding="utf-8")
        sources, meetings = {}, {}
        chunk = dict(self.chunk, turn_ids=[1, 2, 3], _query_terms=["curvy", "texture", "colour"])
        self.agent([]).add_sources([chunk], meetings, sources, "ES2002", "d")
        self.assertIn(("ES2002b", 3), sources)

    def test_internal_coverage_allows_four_verified_role_introductions(self):

        texts = [f"Speaker {speaker} is responsible for the named project role." for speaker in "ABCD"]
        text, turns = "", []
        for number, original in enumerate(texts, 1):
            start = len(text)
            text += original + "\n"
            turns.append({"utterance_id": number, "speaker_id": "ABCD"[number - 1], "timestamp": "00:01:00",
                          "original_text": original, "start_char": start, "end_char": start + len(original)})
        (self.root / "ES2002b.json").write_text(json.dumps(dict(self.meeting,
            original_transcript_text=text, utterances=turns)), encoding="utf-8")
        agent = self.agent([])
        meetings, sources = {}, {}
        agent.add_sources([dict(self.chunk, turn_ids=[1, 2, 3, 4])], meetings, sources, "ES2002", "d")
        part = {"id": "p1", "question": "Who are the four participants?", "requirements": [
            {"id": f"p1_r{number}", "question": f"Role {number}?"} for number in range(1, 5)]}
        response = {"parts": [{"id": "p1", "requirements": [
            {"id": f"p1_r{number}", "evidence": [{"citation_id": f"ES2002b_t{number}_w0"}]}
            for number in range(1, 5)], "next_query": None}]}
        checked, pending = agent.check_coverage(response, [part], meetings, sources, "ES2002", "d")
        self.assertFalse(pending)
        self.assertEqual(len(checked["parts"][0]["evidence"]), 4)
        draft = dict(self.draft, citations=[dict(meeting_id="ES2002b", turn_id=number,
                     quote=original, part_ids=["p1"]) for number, original in enumerate(texts, 1)])
        with self.assertRaisesRegex(ValueError, "one to three citations"):
            agent.validate_draft(draft, {"parts": [part]}, meetings, sources, "ES2002", "d")

    def test_budget_trace_identifies_skipped_turns_and_admitted_sources(self):

        long_text = "word " * 1200
        self.meeting["original_transcript_text"] += "\n" + long_text
        self.meeting["utterances"].append({"utterance_id": 3, "speaker_id": "C", "timestamp": "00:02:00",
            "original_text": long_text, "start_char": len(self.first) + len(self.second) + 2,
            "end_char": len(self.meeting["original_transcript_text"])})
        (self.root / "ES2002b.json").write_text(json.dumps(self.meeting), encoding="utf-8")
        self.search.return_value = [dict(self.chunk, turn_ids=[1, 3])]
        events = []
        result = self.agent([self.plan, self.coverage, self.draft, self.review], on_event=events.append).answer("Power?", "ES2002")
        admission = result["trace"][0]["admission"]
        self.assertEqual(admission["admitted"], [{"meeting_id": "ES2002b", "turn_id": 1}])
        self.assertEqual(admission["dropped"][0]["turn_id"], 3)
        self.assertEqual(admission["dropped"][0]["reason"], "word_budget")
        self.assertLessEqual(admission["words_after"], 1000)
        event = next(event for event in events if event["stage"] == "retrieved")
        self.assertEqual(event["details"]["admission"], admission)

    def test_all_teams_accepts_sources_from_several_teams(self):

        other = dict(self.meeting, meeting_id="TS3012a", team_id="TS3012", meeting_sequence="a", meeting_number=1)
        (self.root / "TS3012a.json").write_text(json.dumps(other), encoding="utf-8")
        self.search.return_value = [self.chunk, dict(self.chunk, chunk_id="TS3012a_c001",
                                                     meeting_id="TS3012a", team_id="TS3012")]
        result = self.agent([self.plan, self.coverage, self.draft, self.review]).answer("Which power source?", "all")
        self.assertEqual(result["status"], "answered")
        self.search.assert_called_once_with("final power source", "ALL", as_of="d", limit=5)
        payload = self.client.generate_json.call_args_list[1].args[1]
        self.assertEqual({source["meeting_id"] for source in payload["sources"]}, {"ES2002b", "TS3012a"})
        self.assertIn("All teams", payload["scope"])

    def test_named_question_team_overrides_all_in_plan_search_and_citations(self):

        agent = self.agent([self.plan, self.coverage, self.draft, self.review])
        result = agent.answer("What power source did ES2002 choose?", "all")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(self.client.generate_json.call_args_list[0].args[1]["team_id"], "ES2002")
        self.search.assert_called_once_with("final power source", "ES2002", as_of="d", limit=5)

    def test_named_question_rejects_other_team_even_if_retriever_returns_it(self):

        self.search.return_value = [dict(self.chunk, meeting_id="IS1008b", team_id="IS1008")]
        with self.assertRaisesRegex(ValueError, "different team"):
            self.agent([self.plan]).answer("ES2002 functional design requirements?", "all")

    def test_all_teams_still_rejects_mismatched_team_metadata(self):

        self.search.return_value = [dict(self.chunk, team_id="TS3012")]
        with self.assertRaises(ValueError):
            self.agent([self.plan]).answer("Which power source?", "ALL")

    def test_progress_callback_observes_real_steps_without_changing_answer(self):

        events = []
        agent = self.agent([self.plan, self.coverage, self.draft, self.review], on_event=events.append)
        result = agent.answer("Which power source?", "ES2002")
        self.assertEqual(result["status"], "answered")
        stages = [event["stage"] for event in events]
        for stage in ("planning", "plan_ready", "searching", "retrieved", "sources_ready",
                      "checking_coverage", "coverage_checked", "drafting", "verifying_answer", "answer_verified"):
            self.assertIn(stage, stages)
        self.assertLess(stages.index("searching"), stages.index("retrieved"))
        self.assertEqual(result["citations"][0]["quote"], self.first)

    def test_multipart_gap_triggers_only_missing_part_search(self):

        self.plan["parts"].append({"question": "Who checks prices?", "query": "price assignment",
                                     "requirements": ["Who owns the price check?"]})
        partial = {"parts": [self.support("p1", 1), self.missing("p2")]}
        complete = {"parts": [self.support("p1", 1), self.support("p2", 2)]}
        draft = {"answer": "The team chose replaceable batteries; marketing will check prices.",
                 "citations": [dict(self.evidence(1), part_ids=["p1"]), dict(self.evidence(2), part_ids=["p2"])]}
        self.search.side_effect = [[self.chunk], [self.chunk], [dict(self.chunk, turn_ids=[2])]]
        result = self.agent([self.plan, partial, complete, draft, self.review]).answer("Power and price owner?", "ES2002")
        self.assertEqual(result["status"], "answered")
        self.assertEqual([c.args[0] for c in self.search.call_args_list],
                         ["final power source", "price assignment", "marketing price follow up"])

    def test_missing_evidence_stops_without_drafting(self):

        missing = {"parts": [self.missing("p1")]}
        result = self.agent([self.plan, missing, missing]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertIsNone(result["answer"])
        self.assertEqual(self.client.generate_json.call_count, 3)

    def test_no_sources_stops_before_coverage(self):

        self.search.return_value = []
        result = self.agent([self.plan]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(self.client.generate_json.call_count, 1)

    def test_wrong_team_future_and_path_injection_rejected(self):

        for meeting_id, team, cutoff in (("TS3012b", "TS3012", "d"), ("ES2002d", "ES2002", "c"),
                                         ("../ES2002b", "ES2002", "d")):
            with self.subTest(meeting_id=meeting_id):
                self.search.return_value = [dict(self.chunk, meeting_id=meeting_id, team_id=team)]
                with self.assertRaises(ValueError):
                    self.agent([self.plan]).answer("Power?", "ES2002", cutoff)

    def test_unsupplied_turn_cannot_supply_evidence(self):

        agent = self.agent([])
        meetings, sources = {}, {}
        agent.add_sources([self.chunk], meetings, sources, "ES2002", "d")
        with self.assertRaisesRegex(ValueError, "supplied turn"):
            agent.check_evidence(self.evidence(2), meetings, sources, "ES2002", "d")

    def test_quote_from_different_turn_is_rejected(self):

        agent = self.agent([])
        meetings, sources = {}, {}
        agent.add_sources([dict(self.chunk, turn_ids=[1, 2])], meetings, sources, "ES2002", "d")
        with self.assertRaisesRegex(ValueError, "supplied turn"):
            agent.check_evidence(dict(self.evidence(1), quote=self.second), meetings, sources, "ES2002", "d")

    def test_short_exact_quotes_are_expanded_in_coverage_and_draft(self):

        self.coverage["parts"][0]["requirements"][0]["evidence"][0]["quote"] = "replaceable batteries."
        self.draft["citations"][0]["quote"] = "replaceable batteries."
        result = self.agent([self.plan, self.coverage, self.draft, self.review]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"][0]["quote"], self.first)
        review_payload = self.client.generate_json.call_args_list[-1].args[1]
        self.assertEqual(review_payload["candidate"]["citations"][0]["quote"], self.first)
        self.assertEqual(review_payload["coverage"]["parts"][0]["evidence"][0]["quote"], self.first)
        self.assertEqual(self.client.generate_json.call_count, 4)

    def test_model_selects_excerpt_ids_and_code_supplies_original_quotes(self):

        identifier = "ES2002b_t1_w0"
        coverage = {"parts": [{"id": "p1", "requirements": [
            {"id": "p1_r1", "evidence": [{"citation_id": identifier}]}], "next_query": None}]}
        draft = {"answer": self.draft["answer"], "citations": [{"citation_id": identifier, "part_ids": ["p1"]}]}
        result = self.agent([self.plan, coverage, draft, self.review]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["citations"], [{"meeting_id": "ES2002b", "quote": self.first}])
        payload = self.client.generate_json.call_args_list[1].args[1]
        self.assertEqual(payload["citation_options"][0]["citation_id"], identifier)
        review = self.client.generate_json.call_args_list[-1].args[1]
        self.assertEqual(review["candidate"]["citations"][0]["quote"], self.first)

    def test_unknown_or_unsupplied_excerpt_ids_cannot_supply_quotes(self):

        agent = self.agent([])
        sources, meetings = {}, {}
        agent.add_sources([self.chunk], meetings, sources, "ES2002", "d")
        for identifier in ("ES2002b_t2_w0", "TS3012d_t1_w0", "ES2002b_t1_w999", [], None):
            with self.subTest(identifier=identifier), self.assertRaises(ValueError):
                agent.check_evidence({"citation_id": identifier}, meetings, sources, "ES2002", "d")

    def test_legacy_quote_cannot_combine_adjacent_supplied_turns(self):

        agent = self.agent([])
        meetings, sources = {}, {}
        agent.add_sources([dict(self.chunk, turn_ids=[1, 2])], meetings, sources, "ES2002", "d")
        citation = dict(self.evidence(1), quote=self.first + " " + self.second)
        with self.assertRaisesRegex(ValueError, "supplied turn"):
            agent.check_evidence(citation, meetings, sources, "ES2002", "d")

    def test_excerpt_windows_cover_long_turn_and_preserve_whitespace(self):

        text = "  ".join(f"word{index}" for index in range(151))
        sources = {("ES2002b", 1): {"text": text}, ("ES2002b", 2): {"text": "No."}}
        options = citation_options(sources)
        covered = set()
        for identifier, option in options.items():
            self.assertTrue(identifier.startswith("ES2002b_t1_"))
            self.assertIn(option["quote"], text)
            self.assertTrue(6 <= len(option["quote"].split()) <= 60)
            covered.update(option["quote"].split())
        self.assertEqual(covered, set(text.split()))
        self.assertEqual(options, citation_options(sources))

    def test_bad_coverage_quote_is_repaired_before_drafting(self):

        invalid = deepcopy(self.coverage)
        invalid["parts"][0]["requirements"][0]["evidence"][0]["quote"] = "This invented sentence is not in the transcript."
        result = self.agent([self.plan, invalid, self.coverage, self.draft, self.review]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "answered")
        repair_payload = self.client.generate_json.call_args_list[2].args[1]
        self.assertEqual(repair_payload["previous_coverage"], invalid)
        self.assertIn("ES2002b turn 1", repair_payload["correction"])
        self.assertEqual(self.client.generate_json.call_count, 5)
        self.assertEqual(self.search.call_count, 1)

    def test_two_invalid_coverage_results_return_failure_without_an_answer(self):

        self.coverage["parts"][0]["requirements"][0]["evidence"][0]["turn_id"] = 999
        result = self.agent([self.plan, self.coverage, self.coverage]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertIsNone(result["answer"])
        self.assertEqual(result["citations"], [])
        self.assertEqual(self.client.generate_json.call_count, 3)

    def test_coverage_repair_budget_is_shared_between_search_rounds(self):

        invalid = deepcopy(self.coverage)
        invalid["parts"][0]["requirements"][0]["evidence"][0]["turn_id"] = 999
        missing = {"parts": [self.missing("p1")]}
        result = self.agent([self.plan, invalid, missing, invalid]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "invalid_evidence")
        self.assertEqual(self.client.generate_json.call_count, 4)

    def test_invalid_quote_repaired_once(self):

        invalid = deepcopy(self.draft)
        invalid["citations"][0]["quote"] = "This invented sentence is not found anywhere."
        result = self.agent([self.plan, self.coverage, invalid, self.draft, self.review]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["trace"][-1]["draft_attempt"], 2)

    def test_semantic_failure_requires_repair(self):

        rejected = {"approved": False, "issues": ["Outcome contradicts evidence"]}
        result = self.agent([self.plan, self.coverage, self.draft, rejected,
                             self.draft, self.review]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "answered")
        self.assertEqual(result["trace"][-1]["draft_attempt"], 2)

    def test_three_invalid_drafts_return_failure(self):

        self.draft["citations"][0]["part_ids"] = []
        result = self.agent([self.plan, self.coverage, self.draft, self.draft, self.draft]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "invalid_answer")
        self.assertEqual(result["citations"], [])

    def test_context_budget_and_source_deduplication(self):

        agent = QuestionAnsweringAgent(self.client, self.search, self.root, max_context_words=1000)
        sources, meetings = {}, {}
        chunks = [dict(self.chunk, turn_ids=[1, 2])] * 4
        self.assertEqual(agent.add_sources(chunks, meetings, sources, "ES2002", "d"), 2)
        self.assertEqual(agent.add_sources(chunks, meetings, sources, "ES2002", "d"), 0)
        self.assertLessEqual(sum(len(json.dumps(s).split()) for s in sources.values()), 1000)

    def test_invalid_plans_rejected(self):

        for plan in ({"parts": []}, {"parts": [self.plan["parts"][0]] * 4}, {"parts": [{}]},
                     {"parts": [{"question": "Power?", "query": "power"}]},
                     {"parts": [dict(self.plan["parts"][0], requirements=[])]}):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                self.agent([plan]).answer("Power?", "ES2002")

    def test_invalid_options_and_question_make_no_calls(self):

        with self.assertRaises(ValueError):
            QuestionAnsweringAgent(self.client, max_rounds=4)
        with self.assertRaises(ValueError):
            self.agent([]).answer(" ", "ES2002")
        self.client.generate_json.assert_not_called()

    def test_coverage_cannot_omit_or_repeat_parts(self):

        self.plan["parts"].append({"question": "Who checks prices?", "query": "prices",
                                     "requirements": ["Who owns the price check?"]})
        for coverage in ({"parts": [self.support("p1", 1)]},
                         {"parts": [self.support("p1", 1), self.support("p1", 1)]}):
            with self.subTest(coverage=coverage):
                result = self.agent([self.plan, coverage, coverage]).answer("Power and prices?", "ES2002")
                self.assertEqual(result["status"], "invalid_evidence")

    def test_oversize_source_turn_is_not_sent_to_model(self):

        long_text = "word " * 1200
        self.meeting["original_transcript_text"] = long_text
        turn = dict(self.meeting["utterances"][0], original_text=long_text, start_char=0, end_char=len(long_text))
        self.meeting["utterances"] = [turn]
        (self.root / "ES2002b.json").write_text(json.dumps(self.meeting), encoding="utf-8")
        result = self.agent([self.plan]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(self.client.generate_json.call_count, 1)

    def test_tampered_source_offsets_rejected(self):

        self.meeting["utterances"][0]["start_char"] = -1
        (self.root / "ES2002b.json").write_text(json.dumps(self.meeting), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "offsets"):
            self.agent([self.plan]).answer("Power?", "ES2002")

    def test_four_sentences_cannot_pass_format_validation(self):

        self.draft["answer"] = "One. Two. Three. Four"
        result = self.agent([self.plan, self.coverage, self.draft, self.draft, self.draft]).answer("Power?", "ES2002")
        self.assertEqual(result["status"], "invalid_answer")


if __name__ == "__main__":
    unittest.main()
