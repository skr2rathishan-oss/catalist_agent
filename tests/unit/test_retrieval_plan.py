"""Regression tests for temporal retrieval policy and repeated source evidence."""

import unittest

from app.retrieval_plan import decision_evidence_score, overlaps_evidence, requested_meeting, select_evidence, wants_current_decision


def passage(chunk_id, text, turns, start=0, end=1000):
    """Build evidence with explicit source ranges for selection tests."""

    return {"chunk_id": chunk_id, "meeting_id": chunk_id.split("_")[0],
            "original_text": text, "turn_ids": turns, "start_char": start,
            "end_char": end, "score": 0.03, "meeting_number": 4}


class TestRetrievalPlan(unittest.TestCase):
    def test_requested_phase_is_distinct_from_latest_allowed_meeting(self):

        for question, expected in (("Who attended the kick-off meeting?", 1),
                                   ("ES2002 functional design requirements?", 2),
                                   ("What trend was reported in the conceptual design meeting?", 3),
                                   ("What was chosen in the detailed design meeting?", 4),
                                   ("What happened in ES2002b?", 2),
                                   ("What happened at meeting C?", 3)):
            with self.subTest(question=question):
                self.assertEqual(requested_meeting(question), expected)

    def test_comparisons_and_cutoffs_keep_historical_sources_available(self):

        for question in ("Compare ES2002a and ES2002c", "What changed since meeting b?",
                         "What was agreed before the detailed design meeting?",
                         "What was current as of ES2002c?", "What did they finally choose?",
                         "Compare the functional and conceptual design meetings"):
            with self.subTest(question=question):
                self.assertIsNone(requested_meeting(question))

    def test_final_intent_is_explicit(self):

        for question in ("What did they finally choose?", "What is the current choice?", "What happened in the end?"):
            self.assertTrue(wants_current_decision(question))
        self.assertFalse(wants_current_decision("What did they initially propose?"))
        self.assertFalse(wants_current_decision("Explain battery power"))

    def test_topical_decision_beats_tentative_recap(self):

        recap = passage("ES2002d_c001", "Just to recap last time, hopefully kinetic power and battery.", [1])
        decision = passage("ES2002d_c007", "Have we got battery? Kinetic as well? No. Okay, just battery.", [102])
        terms = {"power", "battery", "kinetic"}
        self.assertGreater(decision_evidence_score(decision, terms), decision_evidence_score(recap, terms))
        results = select_evidence([recap], 1, [recap, decision], terms)
        self.assertEqual(results[0]["chunk_id"], decision["chunk_id"])

    def test_decision_language_must_share_a_turn_with_topic(self):

        item = passage("ES2002d_c001", "00:00:01 Speaker A\nFinally we decided to use plastic.\n\n00:00:02 Speaker B\nHopefully a battery.", [1, 2])
        self.assertLess(decision_evidence_score(item, {"battery"}), 1)

    def test_long_shared_turn_is_deduplicated_by_source_span(self):

        left = passage("ES2002b_c021", "earlier text", [180, 181, 182, 183, 184, 185], 26204, 28212)
        right = passage("ES2002b_c022", "later text", [185, 186, 187, 188], 27391, 29451)
        self.assertTrue(overlaps_evidence(left, right))
        self.assertEqual(len(select_evidence([left, right], 5)), 1)

    def test_small_overlap_and_other_meetings_remain(self):

        left = passage("ES2002b_c001", "one", [1, 2, 3, 4], 0, 1000)
        right = passage("ES2002b_c002", "two", [4, 5, 6, 7], 950, 1950)
        other_meeting = passage("ES2002c_c001", "three", [1, 2, 3, 4], 0, 1000)
        self.assertEqual(len(select_evidence([left, right, other_meeting], 5)), 3)

    def test_followup_keeps_historical_context_and_does_not_invent_matches(self):

        history = passage("ES2002b_c001", "Battery was proposed.", [1])
        current = passage("ES2002d_c001", "We decided to use a battery.", [1])
        unrelated = passage("ES2002d_c002", "We decided on plastic.", [2], 2000, 3000)
        results = select_evidence([history, current], 3, [current, unrelated], {"battery"})
        self.assertEqual([item["chunk_id"] for item in results], [current["chunk_id"], history["chunk_id"]])

    def test_disjoint_fragments_of_a_long_turn_are_not_duplicates(self):

        left = passage("ES2002b_c001", "first part", [1], 0, 1000)
        right = passage("ES2002b_c002", "second part", [1], 1000, 2000)
        self.assertFalse(overlaps_evidence(left, right))
