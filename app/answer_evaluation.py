"""Keep checked-answer scoring separate from retrieval and agent inputs."""

from datetime import datetime, timezone
import json
from pathlib import Path

from app.config import EVAL_RUNS_DIR, project_path


def summarise_answers(rows: list[dict]) -> dict:
    """Count every case, including abstentions and errors, in the score."""

    counts = {name: sum(row["verdict"] == name for row in rows)
              for name in ("correct", "partial", "incorrect", "no_answer", "error", "not_judged")}
    score = sum(1 if row["verdict"] == "correct" else
                0.5 if row["verdict"] == "partial" else 0 for row in rows)
    return {"questions": len(rows), "counts": counts,
            "answer_score": score / len(rows) if rows and not counts["not_judged"] else None}


def save_answer_report(rows: list[dict], model: str, method: str,
                       runs_dir=EVAL_RUNS_DIR, summary_path=None) -> tuple[Path, Path]:
    """Save local JSON detail and a readable report without changing source cases."""

    runs_dir = project_path(runs_dir)
    summary_path = project_path(summary_path or "evaluation/answer_results.md")
    runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    result = {"time_utc": datetime.now(timezone.utc).isoformat(), "model": model,
              "method": method, "summary": summarise_answers(rows), "questions": rows,
              "review_status": "Model verdicts require human review"}
    path = runs_dir / f"checked_answers_{stamp}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Checked answer evaluation", "",
             f"Run: `{path.name}` · model: `{model}` · method: `{method}`",
             "Model verdicts are provisional until compared with the transcripts by a person.", "",
             f"Questions: {len(rows)} · score: {result['summary']['answer_score']}", "",
             "| ID | Status | Verdict | Reason |", "| --- | --- | --- | --- |"]
    for row in rows:
        reason = (row.get("reason") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {row['question_id']} | {row['status']} | {row['verdict']} | {reason} |")
    lines.extend(["", "## Three failures to review", ""])
    failures = [row for row in rows if row["verdict"] != "correct"]
    if not failures:
        lines.append("No failures in this run; do not invent any.")
    for row in failures[:3]:
        lines.extend([f"### {row['question_id']}", "",
                      f"- Question: {row['question']}",
                      f"- Expected: {row['checked_answer']}",
                      f"- Actual: {row.get('answer') or '(no answer)'}",
                      f"- Model verdict: {row['verdict']}; {row.get('reason') or 'No reason recorded.'}",
                      "- Human source review and root cause: pending.", ""])
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path, summary_path
