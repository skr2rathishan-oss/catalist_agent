"""Validation and atomic persistence for the build-test submission."""

import json
from pathlib import Path
import re

from app.citations import verify_quote
from app.config import ANSWERS_FILE, PARSED_DIR, PROJECT_ROOT, project_path

SUBMISSION_QUESTIONS_FILE = PROJECT_ROOT / "tests" / "questions.json"


def load_submission_questions(path=SUBMISSION_QUESTIONS_FILE) -> list[dict]:
    """Load the official nested question format without answer-side data."""

    data = json.loads(project_path(path).read_text(encoding="utf-8-sig"))
    questions = data.get("questions") if isinstance(data, dict) else None
    if not isinstance(questions, list) or not questions:
        raise ValueError("Submission questions must be a nonempty list")
    seen, checked = set(), []
    for item in questions:
        if not isinstance(item, dict) or set(item) != {"question_id", "team", "question"}:
            raise ValueError("Each submission question needs question_id, team and question")
        identifier, team, question = item["question_id"], item["team"], item["question"]
        if (not isinstance(identifier, str) or identifier in seen
                or not isinstance(team, str) or not isinstance(question, str) or not question.strip()):
            raise ValueError("Submission question contains invalid or duplicate fields")
        seen.add(identifier)
        checked.append({"question_id": identifier, "team": team, "question": question.strip()})
    return checked


def write_submission(answers, brief="", path=ANSWERS_FILE) -> Path:
    """Atomically write exactly the requested public submission fields."""

    if not isinstance(answers, list) or not isinstance(brief, str):
        raise ValueError("Submission requires an answers list and plain-text brief")
    public = []
    for item in answers:
        if not isinstance(item, dict) or set(item) != {"question_id", "answer", "citations"}:
            raise ValueError("Invalid public answer shape")
        public.append(item)
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({"answers": public, "brief": brief}, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    return path


def audit_submission(path=ANSWERS_FILE, questions_path=SUBMISSION_QUESTIONS_FILE,
                     parsed_dir=PARSED_DIR) -> dict:
    """Check saved answer/brief structure and exact local source quotes offline."""

    data = json.loads(project_path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"answers", "brief"}:
        raise ValueError("Submission must contain exactly answers and brief")
    questions = load_submission_questions(questions_path)
    expected = {item["question_id"]: item for item in questions}
    answers = data["answers"]
    if not isinstance(answers, list) or len(answers) != len(expected):
        raise ValueError("Submission answer count does not match the questions")
    meetings = {}

    def checked_quote(meeting_id, quote):
        if not isinstance(meeting_id, str) or not re.fullmatch(r"[A-Z]{2}\d{4}[a-d]", meeting_id):
            raise ValueError("Citation has an invalid meeting ID")
        if not isinstance(quote, str) or not quote.strip():
            raise ValueError("Citation has no quote")
        if meeting_id not in meetings:
            source = project_path(parsed_dir) / f"{meeting_id}.json"
            meetings[meeting_id] = json.loads(source.read_text(encoding="utf-8"))
        meeting = meetings[meeting_id]
        if not any(verify_quote(quote, {"utterances": [turn]}) and
                   meeting["original_transcript_text"][turn["start_char"]:turn["end_char"]]
                   == turn["original_text"] for turn in meeting["utterances"]):
            raise ValueError(f"Quote is not exact in one original turn: {meeting_id}")

    seen, answer_citations = set(), 0
    for answer in answers:
        if not isinstance(answer, dict) or set(answer) != {"question_id", "answer", "citations"}:
            raise ValueError("Invalid saved answer fields")
        identifier = answer["question_id"]
        if identifier not in expected or identifier in seen:
            raise ValueError(f"Unknown or duplicate answer ID: {identifier}")
        seen.add(identifier)
        if not isinstance(answer["answer"], str) or not answer["answer"].strip():
            raise ValueError(f"Answer is empty: {identifier}")
        citations = answer["citations"]
        if not isinstance(citations, list) or not 1 <= len(citations) <= 3:
            raise ValueError(f"Answer needs one to three citations: {identifier}")
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != {"meeting_id", "quote"}:
                raise ValueError(f"Invalid citation fields: {identifier}")
            meeting_id = citation["meeting_id"]
            if (not isinstance(meeting_id, str) or
                    (expected[identifier]["team"].upper() != "ALL" and
                     not meeting_id.startswith(expected[identifier]["team"]))):
                raise ValueError(f"Citation violates question team: {identifier}")
            checked_quote(meeting_id, citation["quote"])
            answer_citations += 1

    from app.agents.brief_agent import BRIEF_SECTIONS
    brief = data["brief"]
    if not isinstance(brief, str) or not brief.strip():
        raise ValueError("Submission brief is empty")
    sections = brief.split("\n\n")
    if len(sections) != len(BRIEF_SECTIONS):
        raise ValueError("Brief section count is incorrect")
    brief_citations = 0
    for section, (heading, _) in zip(sections, BRIEF_SECTIONS):
        if not section.startswith(heading + "\n") or "\nEvidence: " not in section:
            raise ValueError(f"Brief section is malformed: {heading}")
        body, evidence = section.split("\nEvidence: ", 1)
        if not body[len(heading):].strip():
            raise ValueError(f"Brief section is empty: {heading}")
        matches = list(re.finditer(r'\[(TS3012[a-c])\] "(.*?)"(?=; \[TS3012[a-c]\]|$)',
                                   evidence, re.DOTALL))
        if not matches or "; ".join(match.group(0) for match in matches) != evidence:
            raise ValueError(f"Brief evidence is malformed: {heading}")
        for match in matches:
            checked_quote(match.group(1), match.group(2))
            brief_citations += 1
    return {"answers": len(answers), "answer_citations": answer_citations,
            "brief_sections": len(sections), "brief_citations": brief_citations,
            "exact_quote_checks": answer_citations + brief_citations}
