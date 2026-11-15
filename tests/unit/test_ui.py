"""Offline streaming tests; importing the controller does not require Gradio."""

from types import SimpleNamespace
import unittest

from app.generation import GenerationError
from app.llm import EmbeddingError
from app.ui import (alert_html, current_activity, format_reply, parse_backups, process_html, step_cards,
                    stream_question)


class TestUIStreaming(unittest.TestCase):
    def test_process_card_reports_context_budget_drops(self):

        events = [{"stage": "retrieved", "details": {"round": 1, "part_id": "p1",
            "source_turns_added": 1, "chunks": [{"chunk_id": "ES2002c_c001", "meeting_id": "ES2002c",
                                                "original_text": "Original evidence."}],
            "admission": {"words_after": 900, "total_word_budget": 6000,
                          "dropped": [{"meeting_id": "ES2002c", "turn_id": 38, "reason": "word_budget"}]}}}]
        rendered = str(step_cards(events))
        self.assertIn("900 / 6000", rendered)
        self.assertIn("ES2002c turn 38", rendered)
        self.assertIn("skipped by word budget", rendered)

    def factory(self, callback):

        def answer(question, team, cutoff):

            self.assertEqual(cutoff, "d")
            callback({"stage": "planning", "details": {"question": question}})
            callback({"stage": "retrieved", "details": {"source_turns_added": 1, "chunks": [
                {"chunk_id": "ES2002d_c1", "meeting_id": "ES2002d", "original_text": "Original passage"}]}})
            callback({"stage": "sources_ready", "details": {"sources": [{"text": "Original passage"}],
                                                               "citation_options": []}})
            callback({"stage": "coverage_checked", "details": {"missing_parts": []}})
            return {"status": "answered", "answer": "A supported answer.",
                    "citations": [{"meeting_id": "ES2002d", "quote": "Original quoted evidence goes here today."}],
                    "trace": []}

        return SimpleNamespace(answer=answer)

    def test_stream_exposes_progress_evidence_and_final_answer(self):

        updates = list(stream_question("Question?", "ES2002", "all", "bm25", 5, 2, self.factory))
        self.assertEqual(updates[0][0], "Starting")
        self.assertIn("Planning", [row[0] for row in updates])
        self.assertEqual(updates[-1][0], "answered")
        self.assertEqual(updates[-1][1], "A supported answer.")
        self.assertEqual(updates[-1][4]["searches"][0]["chunks"][0]["original_text"], "Original passage")
        self.assertIn("elapsed_seconds", updates[-1][6])
        self.assertEqual(updates[-1][7], "")

    def test_invalid_input_ends_with_clear_error(self):

        def forbidden_factory(callback):
            self.fail("Invalid input must not start an agent")

        updates = list(stream_question("", "ES2002", "d", "bm25", 5, 2, forbidden_factory))
        self.assertEqual(updates[-1][0], "error")
        self.assertEqual(updates[-1][2], [])

    def test_unexpected_errors_do_not_expose_exception_details(self):

        def factory(callback):
            raise RuntimeError("private runtime details")

        updates = list(stream_question("Question?", "ES2002", "d", "bm25", 5, 2, factory))
        self.assertEqual(updates[-1][0], "error")
        self.assertNotIn("private runtime details", str(updates))

    def test_failure_status_is_preserved_and_requests_are_isolated(self):

        def factory(callback):
            return SimpleNamespace(answer=lambda *a: {"status": "invalid_evidence", "answer": None,
                                                      "citations": [], "reason": "Evidence missing", "trace": []})

        list(stream_question("Question?", "ES2002", "d", "bm25", 5, 2, self.factory))
        updates = list(stream_question("Question?", "ES2002", "d", "bm25", 5, 2, factory))
        self.assertEqual(updates[-1][0], "invalid_evidence")
        self.assertEqual(updates[-1][4], {})
        self.assertEqual(updates[-1][2], [])

    def test_rate_limit_is_reported_as_visible_alert(self):

        def factory(callback):
            def answer(*args):
                raise GenerationError("limited", status=429, retry_after="12")
            return SimpleNamespace(answer=answer)

        updates = list(stream_question("Question?", "ES2002", "d", "bm25", 5, 2, factory,
                                       model="m"))
        self.assertEqual(updates[-1][0], "rate_limited")
        self.assertIn("rate limit", updates[-1][1])
        self.assertIn("12 seconds", updates[-1][1])
        self.assertIn('role="alert"', updates[-1][7])
        self.assertIn("Rate limit reached", updates[-1][7])

    def test_embedding_rate_limit_suggests_bm25(self):

        def factory(callback):
            def answer(*args):
                raise EmbeddingError("NVIDIA embedding request failed (429)", status=429)
            return SimpleNamespace(answer=answer)

        updates = list(stream_question("Question?", "ES2002", "d", "hybrid", 5, 2, factory))
        self.assertEqual(updates[-1][0], "rate_limited")
        self.assertIn("bm25", updates[-1][1])

    def test_fallback_and_retry_notices_persist_to_final_answer(self):

        def factory(callback):
            def answer(*args):
                callback({"stage": "rate_limit_retry", "details": {"provider": "nvidia", "model": "model-a",
                          "wait_seconds": 2, "attempt": 1, "max_attempts": 3}})
                callback({"stage": "model_fallback", "details": {"from": {"provider": "nvidia", "model": "model-a"},
                          "to": {"provider": "nvidia", "model": "model-b"}, "reason": "HTTP 429"}})
                return {"status": "answered", "answer": "Done.", "citations": [], "trace": []}
            return SimpleNamespace(answer=answer)

        updates = list(stream_question("Question?", "ES2002", "d", "bm25", 5, 2, factory))
        self.assertEqual(updates[-1][0], "answered")
        self.assertIn("retrying", updates[-1][7])
        self.assertIn("switched to backup", updates[-1][7])
        self.assertIn("continuing with model-b", updates[-1][7])

    def test_alert_html_escapes_text(self):

        self.assertNotIn("<script>", alert_html([{"level": "error", "text": "<script>x</script>"}]))
        self.assertEqual(alert_html([]), "")

    def test_chat_reply_formats_answers_sources_and_limits(self):

        reply = format_reply("answered", "Batteries.", [{"meeting_id": "ES2002d", "quote": "we use batteries"}],
                             {"generation_model": "m2", "routes_used": ["m1", "m2"], "elapsed_seconds": 3})
        self.assertIn("Batteries.", reply)
        self.assertIn("ES2002d", reply)
        self.assertIn("m1 → m2", reply)
        self.assertIn("Rate limit", format_reply("rate_limited", "wait", [], {}))

    def test_steps_render_as_chat_boxes_with_every_chunk(self):

        chunk = {"chunk_id": "ES2002d_c3", "meeting_id": "ES2002d", "score": 0.81234, "original_text":
                 "00:01:02 Speaker A\nthe L_C_D_ is *gone*", "start_timestamp": "00:01:02", "end_timestamp": "00:02:00",
                 "speaker_ids": ["A", "B"], "turn_ids": [4, 9], "retrieval_reason": "Latest available meeting"}
        steps = [{"stage": "model_selected", "details": {"model": "m", "backups": ["b"]}},
                 {"stage": "plan_ready", "details": {"parts": [{"id": "p1", "question": "Q?", "query": "power"}]}},
                 {"stage": "searching", "details": {"round": 1, "part_id": "p1", "query": "power"}},
                 {"stage": "retrieved", "details": {"round": 1, "part_id": "p1", "source_turns_added": 2,
                                                    "chunks": [chunk, dict(chunk, chunk_id="ES2002b_c1",
                                                                           meeting_id="ES2002b")]}},
                 {"stage": "sources_ready", "details": {"sources": [{"meeting_id": "ES2002b"}, {"meeting_id": "ES2002d"}],
                                                        "citation_options": [1, 2]}},
                 {"stage": "coverage_checked", "details": {"round": 1, "missing_parts": [], "coverage": {"parts": [
                     {"id": "p1", "supported": True, "next_query": None,
                      "evidence": [{"meeting_id": "ES2002d", "turn_id": 4, "quote": "the battery stays"}]}]}}}]
        cards = step_cards(steps)
        titles = [title for title, _, _ in cards]
        self.assertEqual(titles[0], "🤖 Model selected")
        self.assertTrue(titles[1].startswith("🧭 Planning"))
        title, search, _ = cards[2]
        self.assertIn("“power”", title)
        self.assertIn("2 chunk(s) retrieved", search)
        self.assertIn("ES2002d (team ES2002, meeting d)", search)
        self.assertIn("ES2002b_c1", search)
        self.assertIn("⏱ 00:01:02–00:02:00", search)
        self.assertIn("score 0.812", search)
        self.assertIn("Why retrieved: Latest available meeting", search)
        self.assertIn("the L_C_D_ is *gone*", search)            # original transcript text, unaltered
        self.assertNotIn("<script>", step_cards([dict(steps[3], details=dict(steps[3]["details"], chunks=[
            dict(chunk, original_text="<script>x</script>")]))])[0][1])
        self.assertIn('latest meeting</span>', cards[3][1])
        self.assertIn("ES2002d (team ES2002, meeting d): 1 turn(s)", cards[3][1])
        self.assertIn("every part supported", cards[4][0])
        panel = process_html(steps[:3], running=True)
        self.assertEqual(panel.count('class="rag-card'), 3)        # 2 finished steps + live "NOW" card
        self.assertIn("Searching: “power”", panel)
        done = process_html(steps, outcome=("answered", "Batteries."))
        self.assertIn("STEP 5", done)
        self.assertIn("Finished — answer verified", done)
        self.assertIn("step by step", process_html([]))
        self.assertEqual(current_activity([]), "🧭 Planning the question…")

    def test_reply_strips_internal_citation_ids(self):

        reply = format_reply("answered", "Fruit trend [ES2002c_t38_w28].", [{"meeting_id": "ES2002c", "quote": "q"}], {})
        self.assertIn("Fruit trend.", reply)
        self.assertNotIn("_t38_", reply)

    def test_backups_are_distinct_models_other_than_primary(self):

        self.assertEqual(parse_backups(["b", "a", "b", "c", "d"], "a"), ["b", "c"])
        self.assertEqual(parse_backups(None, "a"), [])
        with self.assertRaises(ValueError):
            parse_backups([""], "a")

if __name__ == "__main__":
    unittest.main()
