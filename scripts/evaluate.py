"""Score checked retrieval questions; API-backed methods are explicit opt-ins."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import EVAL_QUESTIONS_FILE
from app.evaluation import evaluate_retrieval, load_checked_questions, save_retrieval_report
from app.retriever import search_hybrid, search_keywords
from app.vector_store import search_vectors
from app.llm import NvidiaEmbeddingClient


def main():
    """Run BM25 by default; --method all compares all three retrievers."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bm25", "embedding", "hybrid", "all"), default="bm25")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--questions", default=str(EVAL_QUESTIONS_FILE))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    questions = load_checked_questions(args.questions)
    methods = ("bm25", "embedding", "hybrid") if args.method == "all" else (args.method,)
    client = NvidiaEmbeddingClient() if args.method != "bm25" else None
    functions = {
        "bm25": search_keywords,
        "embedding": lambda *values, **filters: search_vectors(*values, **filters, client=client),
        "hybrid": lambda *values, **filters: search_hybrid(*values, **filters, client=client),
    }
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "limit": args.limit,
              "question_file": str(args.questions), "scope": "retrieval_only", "methods": {}}
    if client:
        report["embedding_model"] = client.settings.model
        report["embedding_dimensions"] = client.settings.dimensions
    for method in methods:
        print(f"Evaluating {method}: {len(questions)} questions at top {args.limit}", flush=True)
        report["methods"][method] = evaluate_retrieval(questions, functions[method], args.limit)
    if client:
        report["api_requests"] = client.api_requests
        report["cache_hits"] = client.cache_hits
    path = save_retrieval_report(report)
    print(json.dumps({"report": str(path), "methods": {name: run["summary"] for name, run in report["methods"].items()}}, indent=2))
    if any(run["summary"]["errors"] or run["summary"]["filter_violations"] for run in report["methods"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
