"""Compose a pre-final-meeting brief from independently grounded QA runs."""

from app.agents.qa_agent import QuestionAnsweringAgent


BRIEF_SECTIONS = (
    ("Decisions already made",
     "Before Team TS3012's final meeting, what concrete product decisions had the team agreed by the end of meeting c? Cover the main design and market choices; distinguish established decisions from proposals."),
    ("What changed",
     "Across Team TS3012 meetings a through c, which important decisions or plans changed? State the earlier and later positions only when both have source evidence."),
    ("Open actions and owners",
     "At the end of Team TS3012 meeting c, which concrete assignments remain open, what deliverables are due, and who owns each? Name an owner only when the source resolves the addressee. Mark an uncertain owner as unknown; do not infer completion from elapsed time."),
    ("Open disagreements",
     "At the end of Team TS3012 meeting c, which product disagreements remain unresolved? Distinguish open objections from topics explicitly settled by discussion or vote. If none are evidenced as open, say so without inventing an issue."),
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
