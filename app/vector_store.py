"""Store NVIDIA vectors locally and search only the requested team's evidence."""

import json
import logging
from pathlib import Path
import sqlite3
import struct
import tempfile
import time

from app.config import BM25_INDEX_FILE, VECTOR_INDEX_FILE, RAW_DIR, SEARCH_LIMIT, project_path
from app.embeddings import FORMAT_VERSION, normalise_vector
from app.index import INDEX_VERSION

VECTOR_VERSION = 1
logger = logging.getLogger(__name__)


def open_readonly(path):
    """Open an existing database without creating an empty one accidentally."""

    path = project_path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Index not found: {path}; build it first")
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def index_fingerprint(connection):
    """Read the build-time fingerprint without scanning the corpus per query."""

    if connection.execute("PRAGMA user_version").fetchone()[0] != INDEX_VERSION:
        raise ValueError("Rebuild the BM25 index before embedding or hybrid search")
    row = connection.execute("SELECT value FROM metadata WHERE name = 'source_fingerprint'").fetchone()
    if not row:
        raise ValueError("Missing source fingerprint; rebuild the BM25 index")
    return row[0]


def build_vector_index(client, keyword_index=BM25_INDEX_FILE, index_path=VECTOR_INDEX_FILE):
    """Embed current indexed chunks; failed builds keep the last good vector index."""

    started = time.perf_counter()
    index_path = project_path(index_path).resolve()
    keyword_index = project_path(keyword_index).resolve()
    if index_path == keyword_index:
        raise ValueError("Keyword and vector indexes must use separate files")
    if index_path.suffix not in (".sqlite3", ".sqlite", ".db") or index_path.is_relative_to(RAW_DIR.parent.resolve()):
        raise ValueError("Vector index must be a SQLite file outside raw inputs")
    source = open_readonly(keyword_index)
    try:
        if source.execute("PRAGMA user_version").fetchone()[0] != INDEX_VERSION:
            raise ValueError("Rebuild the BM25 index before embedding")
        # One read transaction gives every batch the same corpus snapshot.
        source.execute("BEGIN")
        count = source.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if count == 0:
            raise ValueError("Cannot embed an empty keyword index")
        fingerprint = index_fingerprint(source)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="vector-build-", dir=index_path.parent) as directory:
            temporary = Path(directory) / "vectors.sqlite3"
            target = sqlite3.connect(temporary)
            try:
                target.execute("CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
                target.execute(
                    "CREATE TABLE vectors (chunk_id TEXT PRIMARY KEY, team_id TEXT NOT NULL, "
                    "meeting_number INTEGER NOT NULL, payload TEXT NOT NULL, vector BLOB NOT NULL)"
                )
                target.execute("CREATE INDEX vectors_team_order ON vectors(team_id, meeting_number)")
                metadata = {
                    "model": client.settings.model, "dimensions": str(client.settings.dimensions),
                    "format_version": str(FORMAT_VERSION), "source_fingerprint": fingerprint,
                }
                target.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
                cursor = source.execute("SELECT payload FROM chunks ORDER BY chunk_id")
                processed = 0
                while True:
                    rows = cursor.fetchmany(client.settings.batch_size)
                    if not rows:
                        break
                    chunks = [json.loads(row[0]) for row in rows]
                    # Keep natural wording for semantic meaning. Raw evidence is
                    # retained in payload; keyword stop-word stripping is not used.
                    texts = [chunk["normalised_text"] for chunk in chunks]
                    embeddings = client.embed(texts, task="document")
                    if len(embeddings) != len(chunks):
                        raise ValueError("Embedding count does not match chunk count")
                    for chunk, vector in zip(chunks, embeddings):
                        vector = normalise_vector(vector, client.settings.dimensions)
                        packed = struct.pack(f"<{len(vector)}f", *vector)
                        target.execute("INSERT INTO vectors VALUES (?, ?, ?, ?, ?)", (
                            chunk["chunk_id"], chunk["team_id"], chunk["meeting_number"],
                            json.dumps(chunk, ensure_ascii=False), packed,
                        ))
                    processed += len(chunks)
                    logger.info("Vector build: %s/%s chunks", processed, count)
                target.execute(f"PRAGMA user_version = {VECTOR_VERSION}")
                target.commit()
            finally:
                target.close()
            temporary.replace(index_path)
        return {"chunks": count, "model": client.settings.model, "dimensions": client.settings.dimensions,
                "seconds": round(time.perf_counter() - started, 3)}
    finally:
        source.close()


def search_vectors(query, team_id, as_of="d", limit=SEARCH_LIMIT, client=None,
                   index_path=VECTOR_INDEX_FILE, min_score=None, meeting_number=None):
    """Return exact cosine matches after SQL team/cutoff filtering.

    This scans only the selected team's vectors, not every meeting's vectors.
    Scores measure similarity, not claim correctness. Calibrate min_score on
    checked questions rather than choosing an untested confidence threshold.
    """

    from app.retriever import validate_search, validate_meeting_filter
    from app.llm import NvidiaEmbeddingClient

    team_id = validate_search(query, team_id, as_of, limit)
    validate_meeting_filter(meeting_number, as_of)
    if not query.strip():
        return []
    if min_score is not None and not -1 <= min_score <= 1:
        raise ValueError("min_score must be between -1 and 1")
    connection = open_readonly(index_path)
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != VECTOR_VERSION:
            raise ValueError("Unsupported vector index; rebuild embeddings")
        metadata = dict(connection.execute("SELECT name, value FROM metadata"))
        client = client or NvidiaEmbeddingClient()
        if (metadata.get("model") != client.settings.model
                or metadata.get("dimensions") != str(client.settings.dimensions)
                or metadata.get("format_version") != str(FORMAT_VERSION)):
            raise ValueError("Embedding model, dimensions, or format changed; rebuild embeddings")
        from app.config import ALL_TEAMS
        team_filter = "" if team_id == ALL_TEAMS else "team_id = ? AND "
        meeting_filter = " AND meeting_number = ?" if meeting_number is not None else ""
        parameters = ([] if team_id == ALL_TEAMS else [team_id]) + ["abcd".index(as_of) + 1]
        if meeting_number is not None:
            parameters.append(meeting_number)
        rows = connection.execute(
            "SELECT payload, vector FROM vectors WHERE " + team_filter + "meeting_number <= ?" + meeting_filter + " ORDER BY chunk_id",
            parameters,
        ).fetchall()
        if not rows:
            return []  # Do not spend a query call on a team with no eligible data.
        query_vector = normalise_vector(client.embed([query], task="query")[0], client.settings.dimensions)
        results = []
        for payload, packed in rows:
            if len(packed) != client.settings.dimensions * 4:
                raise ValueError("Stored vector is damaged; rebuild embeddings")
            values = list(struct.unpack(f"<{client.settings.dimensions}f", packed))
            vector = normalise_vector(values, client.settings.dimensions)
            score = max(-1.0, min(1.0, sum(a * b for a, b in zip(query_vector, vector))))
            if min_score is not None and score < min_score:
                continue
            result = json.loads(payload)
            result["score"] = score
            results.append(result)
        results.sort(key=lambda result: (-result["score"], result["meeting_number"], result["chunk_id"]))
        return results[:limit]
    finally:
        connection.close()
