"""Build ingestion and BM25; optionally generate cached NVIDIA embeddings."""

import json
from pathlib import Path
import sys
import argparse
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingest import build_ingestion
from app.config import BUILD_LOG_FILE
from app.index import build_keyword_index


def main() -> None:
    """Build all deterministic stages, or rebuild only the keyword index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-only", action="store_true", help="Index existing chunk files without rerunning ingestion")
    parser.add_argument("--embeddings", action="store_true", help="Generate cached NVIDIA embeddings after keyword indexing (uses API quota)")
    args = parser.parse_args()
    BUILD_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(BUILD_LOG_FILE, encoding="utf-8")],
        force=True,
    )
    counts = {}
    if not args.index_only:
        counts["ingestion"] = build_ingestion()
    counts["bm25"] = build_keyword_index()
    if args.embeddings:
        from app.llm import NvidiaEmbeddingClient
        from app.vector_store import build_vector_index
        counts["embeddings"] = build_vector_index(NvidiaEmbeddingClient())
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
