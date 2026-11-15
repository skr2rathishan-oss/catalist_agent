"""Answer one question with a bounded, cached evidence-search loop."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agents.qa_agent import QuestionAnsweringAgent
from app.agents.prompts import GENERATION_CHECK_PROMPT
from app.config import PROJECT_ROOT, ENV_FILE, load_generation_settings
import app.generation as generation_module
from app.generation import GenerationError, NvidiaGenerationClient
from app.llm import EmbeddingError, NvidiaEmbeddingClient
from app.retriever import search_hybrid, search_keywords
from app.vector_store import search_vectors


def main():
    """Print one diagnostic result; leave the final submission untouched."""

    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--question")
    mode.add_argument("--check-generation", action="store_true", help="Check model catalog and one small cached JSON generation")
    mode.add_argument("--diagnostics", action="store_true", help="Show loaded paths and effective generation settings; no API calls")
    parser.add_argument("--team")
    parser.add_argument("--generation-model", help="Explicitly override GENERATION_MODEL for this run")
    parser.add_argument("--as-of", choices=list("abcd"), default="d")
    parser.add_argument("--method", choices=("bm25", "embedding", "hybrid"), default="hybrid")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    if args.question is not None and not args.team:
        parser.error("--team is required with --question")
    try:
        settings = load_generation_settings()
        if args.generation_model:
            settings = replace(settings, model=args.generation_model)
        if args.diagnostics:
            print(json.dumps({
                "project_root": str(PROJECT_ROOT), "python_executable": sys.executable,
                "run_script": str(Path(__file__).resolve()),
                "generation_module": str(Path(generation_module.__file__).resolve()),
                "env_file": str(ENV_FILE), "generation_model": settings.model,
                "generation_provider": settings.provider,
                "process_model_override": os.environ.get("GENERATION_MODEL"),
                "api_key_configured": bool(settings.api_key),
            }, indent=2))
            return
        print(f"Generation model: {settings.model}", file=sys.stderr)
        generation = NvidiaGenerationClient(settings)
        if args.check_generation:
            available = generation.available_models()
            if settings.model not in available:
                raise GenerationError(f"Generation model '{settings.model}' is absent from the authenticated catalog; "
                                      "choose another with --generation-model or GENERATION_MODEL in .env")
            response = generation.generate_json(GENERATION_CHECK_PROMPT, {"operation": "connectivity_check", "version": 1})
            if set(response) != {"ok"} or response["ok"] is not True:
                raise GenerationError("Generation check returned unexpected JSON")
            print(json.dumps({"status": "ok", "generation_model": settings.model,
                              "catalog_listed": True, "api_requests": generation.api_requests,
                              "cache_hits": generation.cache_hits}, indent=2))
            return
        embedding = NvidiaEmbeddingClient() if args.method != "bm25" else None
        searches = {"bm25": search_keywords,
                    "embedding": lambda *a, **kw: search_vectors(*a, **kw, client=embedding),
                    "hybrid": lambda *a, **kw: search_hybrid(*a, **kw, client=embedding)}
        agent = QuestionAnsweringAgent(client=generation, search=searches[args.method],
                                       limit=args.limit, max_rounds=args.max_rounds)
        result = agent.answer(args.question, args.team, args.as_of)
        result["generation_model"] = settings.model
        result["generation_provider"] = settings.provider
    except (ValueError, OSError, sqlite3.DatabaseError, GenerationError, EmbeddingError) as error:
        parser.exit(1, f"Answering failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "answered":
        sys.exit(1)


if __name__ == "__main__":
    main()
