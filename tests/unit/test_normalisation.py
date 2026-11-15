"""Check supported search normalisation rules against explicit expected text.

These tests use temporary inputs only. They do not rebuild the project dataset
or change the existing search database.
"""

import json
from pathlib import Path
import tempfile
import unittest

from app.citations import verify_quote
from app.index import build_keyword_index
from app.ingest import build_ingestion, normalise_text
from app.retriever import search_keywords


class TestNormalisation(unittest.TestCase):
    def assert_examples(self, examples):
        """Show the individual input when a normalisation example fails."""
        for original, expected in examples:
            with self.subTest(original=original):
                self.assertEqual(normalise_text(original), expected)

    def test_acronyms(self):
        self.assert_examples([
            ("L_C_D_", "LCD"),
            ("T_V_ D_V_D_ R_S_I_ L_E_D_", "TV DVD RSI LED"),
            ("l_c_d_", "lcd"),
            ("Keep the L_C_D_ screen.", "Keep the LCD screen."),
            ("LCD and ordinary_text", "LCD and ordinary_text"),
        ])

    def test_filler_sounds(self):
        self.assert_examples([
            ("Um we uh need batteries", "we need batteries"),
            ("UM UHH HMMM MM", ""),
            ("ummm keep hmm the mm screen", "keep the screen"),
            ("album summary rubber human", "album summary rubber human"),
        ])

    def test_single_number_words(self):
        self.assert_examples([
            ("zero", "0"), ("one", "1"), ("two", "2"),
            ("three", "3"), ("four", "4"), ("five", "5"),
            ("six", "6"), ("seven", "7"), ("eight", "8"),
            ("nine", "9"), ("ten", "10"), ("eleven", "11"),
            ("twelve", "12"), ("thirteen", "13"), ("fourteen", "14"),
            ("fifteen", "15"), ("sixteen", "16"), ("seventeen", "17"),
            ("eighteen", "18"), ("nineteen", "19"),
            ("twenty", "20"), ("thirty", "30"), ("forty", "40"),
            ("fifty", "50"), ("sixty", "60"), ("seventy", "70"),
            ("eighty", "80"), ("ninety", "90"), ("TWELVE", "12"),
        ])

    def test_compound_numbers(self):
        self.assert_examples([
            ("twenty five", "25"),
            ("twenty-five", "25"),
            ("Twenty Five", "25"),
            ("thirty one", "31"),
            ("forty-two buttons", "42 buttons"),
            ("ninety nine percent", "99 percent"),
        ])

    def test_currency(self):
        self.assert_examples([
            ("twelve Euro fifty", "12.50 EUR"),
            ("twelve euros fifty", "12.50 EUR"),
            ("TWELVE EURO FIFTY", "12.50 EUR"),
            ("five Euro five", "5.05 EUR"),
            ("zero Euro zero", "0.00 EUR"),
            ("Cost: twelve Euro fifty.", "Cost: 12.50 EUR."),
        ])

    def test_halves_and_simple_decimals(self):
        self.assert_examples([
            ("twelve and a half", "12.5"),
            ("twenty five and a half", "25.5"),
            ("twelve point five", "12.5"),
            ("zero point five", "0.5"),
            ("twenty five point five", "25.5"),
            ("12.50 EUR", "12.50 EUR"),
        ])

    def test_whitespace(self):
        self.assert_examples([
            ("  keep   the screen  ", "keep the screen"),
            ("keep\tthe\nscreen\r\nplease", "keep the screen please"),
            ("keep\u00a0the screen", "keep the screen"),
            ("", ""), (" \t\r\n", ""),
        ])

    def test_replies_and_punctuation_remain_meaningful(self):
        self.assert_examples([
            ("No.", "No."),
            ("Yes.", "Yes."),
            ("I don't agree.", "I don't agree."),
            ("Keep it? No, don't!", "Keep it? No, don't!"),
            ("€12.50 — café", "€12.50 — café"),
            ("someone alone stone", "someone alone stone"),
        ])

    def test_ambiguous_number_sequences_are_not_guessed(self):
        self.assert_examples([
            ("twelve fifty", "twelve fifty"),
            ("one two three", "one two three"),
            ("twenty thirty", "twenty thirty"),
        ])

    def test_normalising_supported_examples_twice_is_stable(self):
        for original in (
            "Um we need twenty-five L_C_D_ screens.",
            "twelve Euro fifty", "twelve and a half", "twelve point five",
            "twelve fifty", "No.", "  keep\tthe\nscreen  ",
        ):
            with self.subTest(original=original):
                once = normalise_text(original)
                self.assertEqual(normalise_text(once), once)

    def test_all_rules_together(self):
        original = "  Um we uh need twenty-five L_C_D_ screens at twelve Euro fifty.\nNo.  "
        self.assertEqual(
            normalise_text(original),
            "we need 25 LCD screens at 12.50 EUR. No.",
        )

    def test_normalised_keywords_retrieve_original_citable_evidence(self):
        """Exercise source → parsing → chunks → SQLite → search → citations."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            spoken = "Um we uh need twenty-five L_C_D_ screens at twelve Euro fifty."
            transcript = (
                "Meeting: ES2002a\nTeam: ES2002\nMeeting number: 1 (a)\n"
                "Speakers: Speaker A, Speaker B\n\n"
                f"00:00:01 Speaker A\n{spoken}\n\n"
                "00:00:02 Speaker B\nNo.\n"
            )
            source = raw / "ES2002a.txt"
            original_bytes = transcript.encode("utf-8")
            source.write_bytes(original_bytes)
            parsed_dir, chunks_dir = root / "parsed", root / "chunks"
            build_ingestion(raw, parsed_dir, chunks_dir)
            meeting = json.loads((parsed_dir / "ES2002a.json").read_text(encoding="utf-8"))
            turn = meeting["utterances"][0]
            self.assertEqual(turn["original_text"], spoken)
            self.assertEqual(turn["normalised_text"], "we need 25 LCD screens at 12.50 EUR.")
            self.assertEqual(meeting["utterances"][1]["normalised_text"], "No.")
            self.assertEqual(meeting["original_transcript_text"][turn["start_char"]:turn["end_char"]], spoken)

            index = root / "bm25.sqlite3"
            build_keyword_index(chunks_dir, index)
            for query in ("LCD", "L_C_D_", "twenty-five", "25", "twelve Euro fifty", "12.50"):
                with self.subTest(query=query):
                    results = search_keywords(query, "ES2002", index_path=index)
                    self.assertEqual(len(results), 1)
                    self.assertIn(spoken, results[0]["original_text"])
                    self.assertIn("25 LCD screens at 12.50 EUR.", results[0]["normalised_text"])
                    self.assertIn(turn["utterance_id"], results[0]["turn_ids"])
            lcd_results = search_keywords("LCD", "ES2002", index_path=index)
            for query in ("lcd", "LCD LCD", "What did ES2002 do with the L_C_D_?", "LCD!!!"):
                with self.subTest(query_processing=query):
                    self.assertEqual(search_keywords(query, "ES2002", index_path=index), lcd_results)
            self.assertEqual(
                search_keywords("screen", "ES2002", index_path=index),
                search_keywords("screens", "ES2002", index_path=index),
            )
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertTrue(verify_quote(spoken, meeting))
            self.assertFalse(verify_quote(turn["normalised_text"], meeting))


if __name__ == "__main__":
    unittest.main()
