"""Run the 25 build-test questions sequentially and write outputs/answers.json."""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.brief_agent import BriefAgent
from app.agents.qa_agent import QuestionAnsweringAgent
from app.config import ANSWERS_FILE, MEMORY_DIR, load_generation_settings, project_path
from app.generation import GenerationError
from app.llm import EmbeddingError, NvidiaEmbeddingClient
from app.providers import create_client, GenerationRouter
from app.retriever import search_hybrid, search_keywords
from app.submission import SUBMISSION_QUESTIONS_FILE, load_submission_questions, write_submission
from app.vector_store import search_vectors


def main():
    """Checkpoint successful answers after every question; never hide failures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(SUBMISSION_QUESTIONS_FILE))
    parser.add_argument("--output", default=str(ANSWERS_FILE))
    parser.add_argument("--method", choices=("bm25", "embedding", "hybrid"), default="hybrid")
    parser.add_argument("--model", help="Primary NVIDIA model ID")
    parser.add_argument("--backup", action="append", default=[],
                        help="Backup after exhausted transient/rate-limit failures (max 2)")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--skip-brief", action="store_true")
    parser.add_argument("--rebuild-brief", action="store_true",
                        help="Regenerate the brief while preserving resumed question answers")
    parser.add_argument("--resume", action="store_true", help="Reuse already answered IDs from the output file")
    args = parser.parse_args()
    if not 1 <= args.limit <= 10 or len(args.backup) > 2:
        parser.error("--limit must be 1-10 and at most two --backup values are allowed")

    questions = load_submission_questions(args.questions)
    model = args.model or load_generation_settings().model
    routes = [("nvidia", name) for name in [model, *args.backup]]
    generation = GenerationRouter(routes) if args.backup else create_client("nvidia", model)
    embedding = NvidiaEmbeddingClient() if args.method != "bm25" else None
    searches = {"bm25": search_keywords,
                "embedding": lambda *a, **kw: search_vectors(*a, **kw, client=embedding),
                "hybrid": lambda *a, **kw: search_hybrid(*a, **kw, client=embedding)}
    output = project_path(args.output)
    answers, brief = [], ""
    if args.resume and output.is_file():
        prior = json.loads(output.read_text(encoding="utf-8"))
        answers = prior.get("answers", [])
        brief = prior.get("brief", "")
        if args.rebuild_brief:
            brief = ""
    completed = {item["question_id"] for item in answers}
    failures = []
    print(f"Running {len(questions)} questions sequentially with {args.method} / {model}", flush=True)
    for number, case in enumerate(questions, 1):
        if case["question_id"] in completed:
            print(f"[{number}/{len(questions)}] {case['question_id']} already complete", flush=True)
            continue
        print(f"[{number}/{len(questions)}] {case['question_id']} {case['team']}: {case['question']}", flush=True)
        started = time.monotonic()
        try:
            agent = QuestionAnsweringAgent(generation, searches[args.method], limit=args.limit,
                                            max_rounds=args.max_rounds, memory_dir=MEMORY_DIR)
            result = agent.answer(case["question"], case["team"], "d")
            if result["status"] != "answered":
                failures.append({"question_id": case["question_id"], "status": result["status"],
                                 "team": case["team"], "question": case["question"],
                                 "reason": result.get("reason"), "trace": result.get("trace", [])})
                print(f"  FAILED {result['status']}: {result.get('reason')}", flush=True)
                break
            answers.append({"question_id": case["question_id"], "answer": result["answer"],
                            "citations": result["citations"]})
            write_submission(answers, brief, output)
            print(f"  answered in {time.monotonic() - started:.1f}s; checkpointed", flush=True)
        except (ValueError, OSError, sqlite3.DatabaseError, GenerationError, EmbeddingError) as error:
            failures.append({"question_id": case["question_id"], "status": "error",
                             "team": case["team"], "question": case["question"],
                             "reason": str(error), "provider_status": getattr(error, "status", None)})
            print(f"  ERROR {type(error).__name__}: {error}", flush=True)
            break

    ordering = {case["question_id"]: index for index, case in enumerate(questions)}
    answers.sort(key=lambda item: ordering[item["question_id"]])
    write_submission(answers, brief, output)

    if not failures and not args.skip_brief and not brief:
        print("Building TS3012 pre-final-meeting brief from meetings a-c", flush=True)
        try:
            brief = BriefAgent(QuestionAnsweringAgent(generation, searches[args.method], limit=args.limit,
                                                      max_rounds=args.max_rounds,
                                                      memory_dir=MEMORY_DIR)).build()
            write_submission(answers, brief, output)
        except (ValueError, OSError, sqlite3.DatabaseError, GenerationError, EmbeddingError) as error:
            failures.append({"question_id": "brief", "status": "error",
                             "team": "TS3012", "question": "Pre-final meeting brief",
                             "reason": str(error), "provider_status": getattr(error, "status", None)})
            print(f"  ERROR building brief: {type(error).__name__}: {error}", flush=True)
    report = output.with_name("submission_run.json")
    report.write_text(json.dumps({"model": model, "backups": args.backup, "method": args.method,
                                  "completed": len(answers), "failures": failures},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(answers)}/{len(questions)}. Output: {output}. Report: {report}", flush=True)
    if failures or len(answers) != len(questions) or (not args.skip_brief and not brief):
        sys.exit(1)


if __name__ == "__main__":
    main()
