"""Score the ten locally checked questions end to end; API use needs --run."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.qa_agent import QuestionAnsweringAgent
from app.answer_evaluation import save_answer_report
from app.config import EVAL_QUESTIONS_FILE, MEMORY_DIR, load_generation_settings
from app.evaluation import load_checked_questions
from app.generation import NvidiaGenerationClient
from app.llm import NvidiaEmbeddingClient
from app.rag_evaluation import judge_answer
from app.retriever import search_hybrid, search_keywords
from app.vector_store import search_vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(EVAL_QUESTIONS_FILE))
    parser.add_argument("--method", choices=("bm25", "embedding", "hybrid"), default="hybrid")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--no-judge", action="store_true",
                        help="Collect answers without model correctness verdicts")
    args = parser.parse_args()
    questions = load_checked_questions(args.questions)
    if not args.run:
        print(json.dumps({"dry_run": True, "questions": len(questions),
                          "api_calls": 0, "note": "Use --run to answer and score."}, indent=2))
        return
    settings = load_generation_settings(model=args.model)
    generation = NvidiaGenerationClient(settings)
    embedding = NvidiaEmbeddingClient() if args.method != "bm25" else None
    search = {"bm25": search_keywords,
              "embedding": lambda *a, **kw: search_vectors(*a, **kw, client=embedding),
              "hybrid": lambda *a, **kw: search_hybrid(*a, **kw, client=embedding)}[args.method]
    agent = QuestionAnsweringAgent(generation, search, limit=args.limit, memory_dir=MEMORY_DIR)
    rows = []
    for case in questions:
        try:
            result = agent.answer(case["question"], case["team"], case["as_of"])
            verdict, reason = "no_answer", result.get("reason")
            if result["status"] == "answered":
                if args.no_judge:
                    verdict, reason = "not_judged", "Human review required"
                else:
                    verdict, reason = judge_answer(
                        generation, {"question": case["question"],
                                     "expected_answer": case["checked_answer"]}, result["answer"])
            rows.append({"question_id": case["question_id"], "question": case["question"],
                         "checked_answer": case["checked_answer"], "answer": result.get("answer"),
                         "status": result["status"], "verdict": verdict, "reason": reason,
                         "citations": result.get("citations", []), "trace": result.get("trace", [])})
        except (OSError, ValueError, RuntimeError) as error:
            rows.append({"question_id": case["question_id"], "question": case["question"],
                         "checked_answer": case["checked_answer"], "answer": None,
                         "status": "error", "verdict": "error",
                         "reason": f"{type(error).__name__}: {error}", "citations": []})
        print(f"{case['question_id']}: {rows[-1]['verdict']}", flush=True)
    path, summary = save_answer_report(rows, settings.model, args.method)
    print(json.dumps({"detail": str(path), "summary": str(summary),
                      "generation_api_requests": generation.api_requests,
                      "embedding_api_requests": embedding.api_requests if embedding else 0}, indent=2))


if __name__ == "__main__":
    main()
