"""Validation and atomic persistence for the build-test submission."""

import json
from pathlib import Path

from app.config import ANSWERS_FILE, PROJECT_ROOT, project_path

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
