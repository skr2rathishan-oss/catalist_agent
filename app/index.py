"""Build a persistent SQLite FTS5 keyword index from meeting chunks."""

import json
import hashlib
import logging
from pathlib import Path
import re
import sqlite3
import tempfile
import time

from app.config import BM25_INDEX_FILE, CHUNKS_DIR, RAW_DIR, project_path
from app.ingest import normalise_text

logger = logging.getLogger(__name__)
INDEX_VERSION = 2


def _validate_chunk(chunk: dict, meeting: dict) -> None:
    """Reject incomplete evidence instead of indexing misleading metadata."""
    required = (
        "chunk_id", "meeting_id", "team_id", "meeting_number", "meeting_sequence",
        "original_text", "turn_ids", "speaker_ids", "start_timestamp", "end_timestamp",
    )
    if not isinstance(chunk, dict) or any(key not in chunk for key in required):
        raise ValueError("Chunk is missing required text or source metadata")
    meeting_id = chunk["meeting_id"]
    if not isinstance(meeting_id, str) or not re.fullmatch(r"[A-Z]{2}\d{4}[a-d]", meeting_id):
        raise ValueError(f"Invalid meeting ID: {meeting_id}")
    for field in ("meeting_id", "team_id", "meeting_number", "meeting_sequence"):
        if chunk[field] != meeting.get(field):
            raise ValueError(f"Chunk metadata differs from its meeting: {field}")
    if chunk["team_id"] != meeting_id[:-1] or chunk["meeting_sequence"] != meeting_id[-1]:
        raise ValueError(f"Inconsistent team or sequence: {meeting_id}")
    if type(chunk["meeting_number"]) is not int or chunk["meeting_number"] != "abcd".index(meeting_id[-1]) + 1:
        raise ValueError(f"Invalid meeting order: {meeting_id}")
    if not isinstance(chunk["chunk_id"], str) or not chunk["chunk_id"].startswith(meeting_id + "_"):
        raise ValueError("Chunk ID must belong to its meeting")
    if not isinstance(chunk["original_text"], str) or not chunk["original_text"].strip():
        raise ValueError("Chunk text must not be empty")
    if not isinstance(chunk["turn_ids"], list) or not chunk["turn_ids"] or any(type(turn) is not int or turn < 1 for turn in chunk["turn_ids"]):
        raise ValueError("Chunk must reference positive original utterance IDs")
    if not isinstance(chunk["speaker_ids"], list) or not chunk["speaker_ids"] or any(not isinstance(speaker, str) for speaker in chunk["speaker_ids"]):
        raise ValueError("Chunk must identify its speakers")
    if any(not isinstance(chunk[field], str) for field in ("start_timestamp", "end_timestamp")):
        raise ValueError("Chunk timestamps must be strings")


def build_keyword_index(
    chunks_dir: str | Path = CHUNKS_DIR,
    index_path: str | Path = BM25_INDEX_FILE,
) -> dict[str, int]:
    """Build a new index and replace the previous one only after success."""
    started = time.perf_counter()
    chunks_dir = project_path(chunks_dir).resolve()
    index_path = project_path(index_path).resolve()
    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory not found: {chunks_dir}")
    files = sorted(chunks_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No chunk files found in {chunks_dir}; run ingestion first")
    if index_path.suffix not in (".sqlite3", ".sqlite", ".db") or index_path.is_relative_to(RAW_DIR.parent.resolve()):
        raise ValueError("Index must be a SQLite file outside the read-only raw directory")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    total_chunks = 0
    # The temporary directory contains only this build's files. A failed build
    # leaves any existing index intact; both connections close before replacement.
    with tempfile.TemporaryDirectory(prefix="bm25-build-", dir=index_path.parent) as directory:
        temporary_index = Path(directory) / "index.sqlite3"
        connection = sqlite3.connect(temporary_index)
        try:
            try:
                connection.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(normalised_text, tokenize='porter unicode61')")
            except sqlite3.OperationalError as error:
                if "no such module: fts5" in str(error).lower():
                    raise RuntimeError("This Python SQLite installation needs FTS5 support") from error
                raise
            connection.execute(
                "CREATE TABLE chunks (chunk_id TEXT UNIQUE NOT NULL, team_id TEXT NOT NULL, "
                "meeting_number INTEGER NOT NULL, payload TEXT NOT NULL)"
            )
            connection.execute("CREATE INDEX chunks_team_order ON chunks(team_id, meeting_number)")
            connection.execute("CREATE TABLE meeting_fingerprints (meeting_id TEXT PRIMARY KEY, digest TEXT NOT NULL)")
            for file in files:
                connection.execute("INSERT INTO meeting_fingerprints VALUES (?, ?)",
                                   (file.stem, hashlib.sha256(file.read_bytes()).hexdigest()))
                meeting = json.loads(file.read_text(encoding="utf-8"))
                if not isinstance(meeting, dict) or not isinstance(meeting.get("chunks"), list) or not meeting["chunks"]:
                    raise ValueError(f"Missing or empty chunk list in {file.name}")
                if meeting.get("total_chunks") != len(meeting["chunks"]):
                    raise ValueError(f"Incorrect chunk count in {file.name}")
                for chunk in meeting["chunks"]:
                    _validate_chunk(chunk, meeting)
                    # Refresh the derived copy, including when loading older chunks.
                    chunk["normalised_text"] = normalise_text(chunk["original_text"])
                    row = connection.execute(
                        "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                        (chunk["chunk_id"], chunk["team_id"], chunk["meeting_number"], json.dumps(chunk, ensure_ascii=False)),
                    )
                    connection.execute("INSERT INTO chunks_fts(rowid, normalised_text) VALUES (?, ?)", (row.lastrowid, chunk["normalised_text"]))
                    total_chunks += 1
            digest = hashlib.sha256()
            for row in connection.execute("SELECT payload FROM chunks ORDER BY chunk_id"):
                digest.update(row[0].encode("utf-8"))
                digest.update(b"\n")
            connection.execute("CREATE TABLE metadata (name TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO metadata VALUES ('source_fingerprint', ?)", (digest.hexdigest(),))
            connection.execute(f"PRAGMA user_version = {INDEX_VERSION}")
            connection.commit()
        finally:
            connection.close()
        temporary_index.replace(index_path)
    logger.info("Indexed %s meetings and %s chunks in %.2fs (no model calls)", len(files), total_chunks, time.perf_counter() - started)
    return {"meetings": len(files), "chunks": total_chunks}


def update_keyword_index(chunks_dir: str | Path = CHUNKS_DIR,
                         index_path: str | Path = BM25_INDEX_FILE) -> dict[str, int]:
    """Update only changed meetings; retain the corpus fingerprint used by hybrid search."""

    chunks_dir, index_path = project_path(chunks_dir), project_path(index_path)
    if not index_path.is_file():
        return build_keyword_index(chunks_dir, index_path)
    connection = sqlite3.connect(index_path)
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != INDEX_VERSION:
            raise ValueError("Unsupported keyword index; rebuild it")
        if not connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name='meeting_fingerprints'").fetchone():
            connection.close()
            return build_keyword_index(chunks_dir, index_path)
        files = {path.stem: path for path in chunks_dir.glob("*.json")}
        if not files:
            raise ValueError("No chunk files found")
        current = {meeting_id: hashlib.sha256(path.read_bytes()).hexdigest()
                   for meeting_id, path in files.items()}
        prior = dict(connection.execute("SELECT meeting_id, digest FROM meeting_fingerprints"))
        changed = sorted(set(current) ^ set(prior) |
                         {name for name in current.keys() & prior.keys()
                          if current[name] != prior[name]})
        if changed:
            for meeting_id in changed:
                if not re.fullmatch(r"[A-Z]{2}\d{4}[a-d]", meeting_id):
                    raise ValueError(f"Invalid chunk meeting ID: {meeting_id}")
                rows = connection.execute(
                    "SELECT rowid FROM chunks WHERE team_id=? AND meeting_number=?",
                    (meeting_id[:-1], "abcd".index(meeting_id[-1]) + 1)).fetchall()
                for (rowid,) in rows:
                    connection.execute("DELETE FROM chunks_fts WHERE rowid=?", (rowid,))
                    connection.execute("DELETE FROM chunks WHERE rowid=?", (rowid,))
                connection.execute("DELETE FROM meeting_fingerprints WHERE meeting_id=?", (meeting_id,))
                if meeting_id not in files:
                    continue
                data = json.loads(files[meeting_id].read_text(encoding="utf-8"))
                if data.get("meeting_id") != meeting_id or data.get("total_chunks") != len(data.get("chunks", [])):
                    raise ValueError(f"Invalid chunk file: {meeting_id}")
                for chunk in data["chunks"]:
                    _validate_chunk(chunk, data)
                    chunk["normalised_text"] = normalise_text(chunk["original_text"])
                    cursor = connection.execute(
                        "INSERT INTO chunks VALUES (?, ?, ?, ?)",
                        (chunk["chunk_id"], chunk["team_id"], chunk["meeting_number"],
                         json.dumps(chunk, ensure_ascii=False)))
                    connection.execute("INSERT INTO chunks_fts(rowid, normalised_text) VALUES (?, ?)",
                                       (cursor.lastrowid, chunk["normalised_text"]))
                connection.execute("INSERT INTO meeting_fingerprints VALUES (?, ?)",
                                   (meeting_id, current[meeting_id]))
            digest = hashlib.sha256()
            for (payload,) in connection.execute("SELECT payload FROM chunks ORDER BY chunk_id"):
                digest.update(payload.encode("utf-8"))
                digest.update(b"\n")
            connection.execute("UPDATE metadata SET value=? WHERE name='source_fingerprint'",
                               (digest.hexdigest(),))
            connection.commit()
        chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"meetings": len(files), "chunks": chunks, "updated_meetings": len(changed)}
    finally:
        connection.close()
