"""Evaluate retrieved evidence separately from answer correctness."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time

from app.config import EVAL_QUESTIONS_FILE, EVAL_RESULTS_FILE, EVAL_RUNS_DIR, PARSED_DIR, project_path
from app.citations import verify_quote
from app.retriever import validate_search


def comparable(text):
    """Use the citation contract's case and whitespace equivalence only."""

    return " ".join(text.lower().split())


def load_checked_questions(path=EVAL_QUESTIONS_FILE, parsed_dir=PARSED_DIR):
    """Validate every gold quote and turn reference before scoring retrieval."""

    data = json.loads(project_path(path).read_text(encoding="utf-8-sig"))
    questions = data.get("questions", [])
    if not questions:
        raise ValueError("Evaluation questions are empty")
    seen = set()
    for question in questions:
        identifier = question["question_id"]
        if identifier in seen:
            raise ValueError(f"Duplicate evaluation question: {identifier}")
        seen.add(identifier)
        validate_search(question["question"], question["team"], question["as_of"], 1)
        if not question.get("checked_answer") or not question.get("evidence"):
            raise ValueError(f"Missing checked answer or evidence: {identifier}")
        references = set()
        for evidence in question["evidence"]:
            meeting_id = evidence["meeting_id"]
            allowed = [question["team"] + letter for letter in "abcd"[:"abcd".index(question["as_of"]) + 1]]
            if meeting_id not in allowed:
                raise ValueError(f"Gold evidence violates team/cutoff: {identifier}")
            reference = (meeting_id, evidence["turn_id"])
            if reference in references:
                raise ValueError(f"Duplicate gold evidence: {identifier}")
            references.add(reference)
            meeting = json.loads((project_path(parsed_dir) / f"{meeting_id}.json").read_text(encoding="utf-8"))
            turns = [turn for turn in meeting["utterances"] if turn["utterance_id"] == evidence["turn_id"]]
            if len(turns) != 1 or not verify_quote(evidence["quote"], {"utterances": turns}):
                raise ValueError(f"Gold quote does not match its original turn: {identifier}")
            turn = turns[0]
            if meeting["original_transcript_text"][turn["start_char"]:turn["end_char"]] != turn["original_text"]:
                raise ValueError(f"Invalid gold source offsets: {identifier}")
    return questions


def score_evidence(question, results):
    """Count distinct gold turns, so overlapping chunks cannot inflate recall."""

    ranks = []
    for evidence in question["evidence"]:
        matches = []
        for rank, result in enumerate(results, start=1):
            if (result["team_id"] == question["team"]
                    and result["meeting_id"] == evidence["meeting_id"]
                    and result["meeting_number"] <= "abcd".index(question["as_of"]) + 1
                    and evidence["turn_id"] in result["turn_ids"]
                    and comparable(evidence["quote"]) in comparable(result["original_text"])):
                matches.append(rank)
        ranks.append(min(matches) if matches else None)
    found = [rank for rank in ranks if rank is not None]
    violations = sum(result["team_id"] != question["team"] or result["meeting_number"] > "abcd".index(question["as_of"]) + 1 for result in results)
    return {
        "any_evidence_found": bool(found),
        "all_evidence_found": len(found) == len(ranks),
        "evidence_recall": len(found) / len(ranks),
        "reciprocal_rank": 1 / min(found) if found else 0.0,
        "filter_violations": violations,
        "evidence_ranks": ranks,
    }


def summarise(rows):
    """Include failed searches in denominators instead of silently dropping them."""

    count = len(rows)
    if not count:
        return {"questions": 0}
    return {
        "questions": count,
        "hit_rate": sum(row["any_evidence_found"] for row in rows) / count,
        "complete_evidence_rate": sum(row["all_evidence_found"] for row in rows) / count,
        "mean_evidence_recall": sum(row["evidence_recall"] for row in rows) / count,
        "mrr": sum(row["reciprocal_rank"] for row in rows) / count,
        "filter_violations": sum(row["filter_violations"] for row in rows),
        "errors": sum(row["error"] is not None for row in rows),
    }


def evaluate_retrieval(questions, search, limit=5):
    """Send only question/team/cutoff to retrieval; gold answers stay in scoring."""

    rows = []
    for question in questions:
        started = time.perf_counter()
        error = None
        try:
            results = search(question["question"], question["team"], as_of=question["as_of"], limit=limit)
        except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError) as failure:
            results = []
            error = f"{type(failure).__name__}: {failure}"
        row = {
            "question_id": question["question_id"], "question": question["question"],
            "split": question.get("split", "development"), "team": question["team"],
            "as_of": question["as_of"], "error": error,
            "seconds": round(time.perf_counter() - started, 3),
            "retrieved_chunks": [item["chunk_id"] for item in results],
            "expected_evidence": question["evidence"],
        }
        row.update(score_evidence(question, results))
        rows.append(row)
    splits = {name: summarise([row for row in rows if row["split"] == name]) for name in sorted({row["split"] for row in rows})}
    return {"summary": summarise(rows), "splits": splits, "questions": rows}


def save_retrieval_report(report, runs_dir=EVAL_RUNS_DIR, results_file=EVAL_RESULTS_FILE):
    """Keep full timestamped results and write a concise latest-run summary."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    runs_dir, results_file = project_path(runs_dir), project_path(results_file)
    runs_dir.mkdir(parents=True, exist_ok=True)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"retrieval_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Retrieval evaluation", "", f"UTC run: {stamp}", f"Full results: `{path.name}` in `{runs_dir}`.", "",
             "This measures evidence retrieval, not generated-answer correctness. Source quotes were checked before the run; user review of the question set is pending.", "",
             "| Method | Split | Questions | Any evidence | All evidence | Mean evidence recall | MRR | Filter violations | Errors |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method, run in report["methods"].items():
        groups = {"all": run["summary"], **run["splits"]}
        for name, values in groups.items():
            lines.append(f"| {method} | {name} | {values['questions']} | {values['hit_rate']:.1%} | {values['complete_evidence_rate']:.1%} | {values['mean_evidence_recall']:.1%} | {values['mrr']:.3f} | {values['filter_violations']} | {values['errors']} |")
    for method, run in report["methods"].items():
        lines.extend(["", f"## {method}: missed evidence or failed searches", ""])
        failures = [row for row in run["questions"] if not row["all_evidence_found"] or row["filter_violations"] or row["error"]]
        if not failures:
            lines.append("No missed gold evidence in this run. This small set does not establish general quality.")
        for row in failures:
            lines.append(f"- {row['question_id']}: evidence ranks {row['evidence_ranks']}; chunks {row['retrieved_chunks']}; error {row['error']}; filter violations {row['filter_violations']}.")
    lines.extend(["", "Three answer-level failures and explanations must be added after an answering agent is implemented. Retrieval misses above are not automatically wrong answers."])
    results_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
