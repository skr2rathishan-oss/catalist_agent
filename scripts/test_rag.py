"""Run the RAG agent on tests/rag_test_questions.json and score every answer.

Makes NVIDIA API calls (agent + optional judge). Responses are cached, so a
rerun can reuse matching requests. Changed context/settings and cache misses use quota.

Examples:
  python scripts/test_rag.py                      # all 20 questions, hybrid search, judged
  python scripts/test_rag.py --method bm25        # no embedding calls
  python scripts/test_rag.py --ids 1,7,20         # a few questions
  python scripts/test_rag.py --team-mode all      # test the "All teams" mode
"""

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.qa_agent import QuestionAnsweringAgent
from app.config import ALL_TEAMS, load_generation_settings
from app.generation import GenerationError
from app.llm import EmbeddingError, NvidiaEmbeddingClient
from app.providers import create_client, GenerationRouter
from app.rag_evaluation import (RAG_QUESTIONS_FILE, judge_answer, load_rag_questions, save_rag_report,
                                score_case, summarise)
from app.retriever import search_hybrid, search_keywords
from app.vector_store import search_vectors


def print_answer(case, result, verdict, reason, retrieved, error, show_trace=False, evidence=None):
    """Display reference material only after the independent agent run."""

    print("\nAGENT ANSWER:", flush=True)
    print((result or {}).get("answer") or (result or {}).get("reason") or error or "No answer", flush=True)
    print("\nCITATIONS:", flush=True)
    for citation in (result or {}).get("citations", []):
        print(f"  [{citation['meeting_id']}] {citation['quote']}", flush=True)
    print("\nEXPECTED ANSWER (reference, not supplied to the agent):", flush=True)
    print(case["expected_answer"], flush=True)
    print("Expected source meetings: " + ", ".join(case["meetings"]), flush=True)
    print("Retrieved meetings: " + (", ".join(sorted(set(retrieved))) or "none"), flush=True)
    print(f"Status: {(result or {}).get('status', 'error')} | Verdict: {verdict}", flush=True)
    if reason:
        print("Judge explanation: " + reason, flush=True)
    if show_trace:
        print("\nTRACE:\n" + json.dumps((result or {}).get("trace", []), ensure_ascii=False, indent=2), flush=True)
    if evidence is not None:
        print("\nRETRIEVED PASSAGES:\n" + json.dumps(evidence, ensure_ascii=False, indent=2), flush=True)


def main():

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=str(RAG_QUESTIONS_FILE))
    parser.add_argument("--method", choices=("bm25", "embedding", "hybrid"), default="hybrid")
    parser.add_argument("--team-mode", choices=("auto", "all"), default="auto",
                        help="auto: team from each question's sources (ALL for multi-team); all: always ALL teams")
    parser.add_argument("--ids", "--id", help="Question ID or comma-separated IDs, e.g. 1 or 1,2,20")
    parser.add_argument("--interactive", action="store_true", help="Press Enter before each question; s skips, q quits")
    parser.add_argument("--show-trace", action="store_true", help="Print the agent trace after every answer")
    parser.add_argument("--show-evidence", action="store_true", help="Print the retrieved passages after every answer")
    parser.add_argument("--as-of", choices=list("abcd"), default="d", help="Explicit meeting cutoff, default d (all meetings)")
    parser.add_argument("--limit", type=int, default=5, help="Chunks per search (1-10)")
    parser.add_argument("--max-rounds", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--model", help="NVIDIA generation model (default: GENERATION_MODEL)")
    parser.add_argument("--backup", action="append", default=[],
                        help="Backup NVIDIA model after exhausted transient/rate-limit failures (max 2)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM grading of answers")
    args = parser.parse_args()
    if not 1 <= args.limit <= 10:
        parser.error("--limit must be between 1 and 10")
    if len(args.backup) > 2:
        parser.error("Choose at most two backup models")

    cases = load_rag_questions(args.questions)
    if args.ids:
        try:
            wanted = {int(value) for value in args.ids.split(",") if value.strip()}
        except ValueError:
            parser.error("--ids must contain numeric question IDs")
        unknown = wanted - {case["id"] for case in cases}
        if unknown:
            parser.error(f"Unknown question IDs: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in wanted]
        if not cases:
            parser.error("No questions match --ids")
    if args.team_mode == "all":
        cases = [dict(case, team=ALL_TEAMS) for case in cases]

    model = args.model or load_generation_settings().model
    backups = args.backup
    routes = [("nvidia", name) for name in [model, *backups]]
    # One shared client keeps the configured requests-per-minute spacing across questions.
    generation = GenerationRouter(routes) if backups else create_client("nvidia", model)
    judge = None if args.no_judge else (GenerationRouter(routes) if backups else create_client("nvidia", model))
    embedding = NvidiaEmbeddingClient() if args.method != "bm25" else None
    searches = {"bm25": search_keywords,
                "embedding": lambda *a, **kw: search_vectors(*a, **kw, client=embedding),
                "hybrid": lambda *a, **kw: search_hybrid(*a, **kw, client=embedding)}

    print(f"Testing {len(cases)} questions · method={args.method} · model={model}"
          + (f" · backups={backups}" if backups else "") + f" · judge={'off' if judge is None else 'on'}", flush=True)
    rows = []
    stopped_for_limit = False
    limit_message = ""
    for number, case in enumerate(cases, 1):
        print(f"\n[{number}/{len(cases)}] Q{case['id']} [{case.get('type', 'unknown')}] "
              f"team={case['team']} cutoff={args.as_of}\n{case['question']}", flush=True)
        if args.interactive:
            try:
                choice = input("Enter = run, s = skip, q = quit: ").strip().lower()
                while choice not in ("", "s", "q"):
                    choice = input("Use Enter, s, or q: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nStopped before running this question.")
                break
            if choice == "q":
                break
            if choice == "s":
                continue
        retrieved = []
        passages = []

        def collect(event):
            if event["stage"] == "retrieved":
                retrieved.extend(chunk["meeting_id"] for chunk in event["details"]["chunks"])
                if args.show_evidence:
                    passages.append(event["details"])

        agent = QuestionAnsweringAgent(generation, searches[args.method], max_rounds=args.max_rounds,
                                       limit=args.limit, on_event=collect)
        started = time.monotonic()
        result, error, verdict, reason = None, None, "not_judged", None
        try:
            result = agent.answer(case["question"], case["team"], args.as_of)
            if result["status"] != "answered":
                verdict = "no_answer"
            elif judge is not None:
                verdict, reason = judge_answer(judge, case, result["answer"])
        except (ValueError, OSError, sqlite3.DatabaseError, GenerationError, EmbeddingError) as failure:
            error, verdict = f"{type(failure).__name__}: {failure}", "error"
            stopped_for_limit = getattr(failure, "status", None) in (429, 402)
            limit_message = ("Embedding limit reached. Use --method bm25 to avoid query embeddings, or wait."
                             if isinstance(failure, EmbeddingError) else
                             "All selected model routes reached their limit. Change --model or wait, or add another --backup.")
        row = score_case(case, result, retrieved, verdict, reason, time.monotonic() - started, error)
        row["citations"] = (result or {}).get("citations", [])
        row["trace"] = (result or {}).get("trace", [])
        if args.show_evidence:
            row["retrieved_passages"] = passages
        rows.append(row)
        print_answer(case, result, verdict, reason, retrieved, error, args.show_trace,
                     passages if args.show_evidence else None)
        print(f"[{number}/{len(cases)}] Q{case['id']} {case['team']:<6} {verdict:<10} "
              f"recall={row['retrieval_recall']:.0%} cited={','.join(row['cited_meetings']) or '-'} "
              f"({row['seconds']:.0f}s)" + (f"  {error}" if error else ""), flush=True)
        if stopped_for_limit:
            print(limit_message + " Stopping here and saving completed results.", flush=True)
            break

    if not rows:
        print("No questions were run; no report was written.")
        return

    report = {"question_file": args.questions, "method": args.method, "team_mode": args.team_mode,
              "model": model, "backups": backups, "judge": judge is not None, "limit": args.limit,
              "as_of": args.as_of, "stopped_for_limit": stopped_for_limit,
              "max_rounds": args.max_rounds, "summary": summarise(rows), "questions": rows,
              "generation_requests": generation.api_requests + (judge.api_requests if judge else 0),
              "generation_cache_hits": generation.cache_hits + (judge.cache_hits if judge else 0),
              "embedding_requests": embedding.api_requests if embedding else 0}
    run_file, results_file = save_rag_report(report)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {results_file}\nFull JSON: {run_file}")
    if stopped_for_limit or any(row["error"] for row in rows):
        sys.exit(1)


if __name__ == "__main__":
    main()
