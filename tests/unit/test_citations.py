"""Regression tests for source quotations and meeting boundaries."""

import unittest

from app.citations import quote_from_turn, quote_with_context, verify_citation, verify_quote


class TestCitations(unittest.TestCase):
    def setUp(self):
        self.meeting = {
            "team_id": "TS3012", "meeting_sequence": "c",
            "utterances": [
                {"original_text": "We should keep the L_C_D_ screen, uh yes."},
                {"original_text": "Because that is what the customers wanted."},
            ],
        }

    def test_exact_quote_and_allowed_normalisation(self):
        self.assertTrue(verify_quote("WE should  keep the L_C_D_ screen,", self.meeting))
        self.assertFalse(verify_quote("We should keep the LCD screen, yes.", self.meeting))

    def test_short_and_cross_turn_quotes_fail(self):
        self.assertFalse(verify_quote("keep the screen", self.meeting))
        self.assertFalse(verify_quote("screen, uh yes. Because that is what", self.meeting))

    def test_source_span_and_filters(self):
        quote = quote_from_turn(self.meeting["utterances"][0], max_words=6)
        self.assertTrue(verify_quote(quote, self.meeting))
        citation = {"meeting_id": "TS3012c", "quote": quote}
        meetings = {"TS3012c": self.meeting}
        self.assertTrue(verify_citation(citation, meetings, "TS3012", "c"))
        self.assertFalse(verify_citation(citation, meetings, "TS3012", "b"))
        self.assertFalse(verify_citation(citation, meetings, "ES2002"))
        self.assertFalse(verify_citation({"meeting_id": "missing", "quote": quote}, meetings))

    def test_partial_words_fail(self):
        self.assertFalse(verify_quote("ecause that is what the customers wanted.", self.meeting))

    def test_short_turn_cannot_supply_a_quote(self):
        with self.assertRaises(ValueError):
            quote_from_turn({"original_text": "Yes."})

    def test_short_exact_span_gets_original_context(self):
        text = ("um if we need to rethink anything at all. So um for this first part here power-wise, "
                "have we got battery? Do we have kinetic as well? No. Okay, just battery.")
        turn = {"original_text": text}
        quote = quote_with_context("just battery", turn)
        self.assertEqual(quote, text)
        self.assertGreaterEqual(len(quote.split()), 6)
        self.assertTrue(verify_quote(quote, {"utterances": [turn]}))
        self.assertFalse(verify_quote("just battery", {"utterances": [turn]}))

    def test_context_keeps_original_case_punctuation_and_spacing(self):
        text = "Before  we picked\nthe L_C_D_ screen, uh yes."
        self.assertEqual(quote_with_context("l_c_d_   SCREEN,", {"original_text": text}), text)

    def test_context_uses_bounded_window_containing_the_selected_span(self):
        text = " ".join([f"word{number}" for number in range(60)])
        for position in (0, 30, 58):
            with self.subTest(position=position):
                selected = f"word{position} word{position + 1}"
                quote = quote_with_context(selected, {"original_text": text}, max_words=12)
                self.assertEqual(len(quote.split()), 12)
                self.assertIn(selected, quote)
                self.assertIn(quote, text)

    def test_context_rejects_repeated_or_invented_spans(self):
        text = "The battery design uses a battery pack with cells."
        for quote in ("battery", "rechargeable cells", "attery design", "", None):
            with self.subTest(quote=quote), self.assertRaises(ValueError):
                quote_with_context(quote, {"original_text": text})

    def test_context_does_not_cross_turns_or_change_normalised_wording(self):
        with self.assertRaises(ValueError):
            quote_with_context("just battery", {"original_text": "No. Okay, just battery."})
        with self.assertRaises(ValueError):
            quote_with_context("LCD screen", self.meeting["utterances"][0])

    def test_context_rejects_invalid_word_budget(self):
        for budget in (5, True, 6.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                quote_with_context("L_C_D_ screen,", self.meeting["utterances"][0], budget)
