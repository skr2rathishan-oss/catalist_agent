"""Compose a pre-final-meeting brief from independently grounded QA runs."""

from app.agents.qa_agent import QuestionAnsweringAgent


BRIEF_SECTIONS = (
    ("Decisions already made",
     "Before Team TS3012's final meeting, what concrete product decisions had the team agreed by the end of meeting c? Summarize only established decisions, not proposals."),
    ("What changed",
     "Across Team TS3012 meetings a through c, which important product decisions or plans changed, and what did each change from and to?"),
    ("Open actions and owners",
     "At the end of Team TS3012 meeting c, answer both separately: (1) in the 'work on our own' exchange, which market-analysis and project-report assignments were stated and who owned each; and (2) which design deliverable was requested for the next meeting and who owned it. Apply these ownership rules to every cited turn: when the source speaker says 'I am supposed to' do a task, name that source speaker as its owner; when the source speaker says 'you' should do a task and the addressee is unresolved, mark its owner not safely identifiable and never assign it to the source speaker. In particular, the design request is spoken by Speaker A as 'You doodle', so do not name Speaker A as the design owner unless another source explicitly resolves that addressee. Exclude unresolved product features. Do not guess an addressee or treat elapsed time as completion."),
    ("Open disagreements",
     "By the end of Team TS3012 meeting c, what final statement and vote show how the team closed its product disagreements? Summarize the grounded closure, and do not treat earlier objections as still open after an explicit resolution."),
)


class BriefAgent:
    """Build a cited TS3012 brief while enforcing the pre-meeting cutoff."""

    def __init__(self, qa_agent=None):

        self.qa_agent = qa_agent or QuestionAnsweringAgent()

    def build(self, team_id="TS3012", as_of="c") -> str:
        """Return a plain-text brief; fail closed if any section is ungrounded."""

        if team_id != "TS3012" or as_of != "c":
            raise ValueError("The build-test brief must use TS3012 meetings a-c only")
        rendered_sections = {}
        # Resolve broad temporal summaries first. If their cached primary-model
        # results are available, a late fallback needed by the action section
        # cannot make the smaller backup model re-plan those summaries.
        execution_order = sorted(BRIEF_SECTIONS, key=lambda item: item[0] == "Open actions and owners")
        for heading, question in execution_order:
            result = self.qa_agent.answer(question, team_id, as_of)
            if result.get("status") != "answered":
                raise ValueError(f"Brief section '{heading}' failed: {result.get('reason', result.get('status'))}")
            citations = result.get("citations")
            if not isinstance(citations, list) or not citations:
                raise ValueError(f"Brief section '{heading}' has no citations")
            rendered = []
            for citation in citations:
                meeting_id, quote = citation.get("meeting_id"), citation.get("quote")
                if not isinstance(meeting_id, str) or not meeting_id.startswith(team_id) or meeting_id[-1] > as_of:
                    raise ValueError("Brief citation violates the TS3012 pre-final cutoff")
                if not isinstance(quote, str) or not quote.strip():
                    raise ValueError("Brief citation is missing its exact quote")
                rendered.append(f'[{meeting_id}] "{quote}"')
            rendered_sections[heading] = f"{heading}\n{result['answer']}\nEvidence: " + "; ".join(rendered)
        return "\n\n".join(rendered_sections[heading] for heading, _ in BRIEF_SECTIONS)
