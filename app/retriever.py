"""Keyword, semantic and hybrid retrieval with strict team and meeting filters."""

import argparse
import json
from pathlib import Path
import re
import sqlite3

from app.config import ALL_TEAMS, BM25_INDEX_FILE, VECTOR_INDEX_FILE, SEARCH_LIMIT, project_path
from app.index import INDEX_VERSION
from app.ingest import normalise_text

STOP_WORDS = set("a an the and or of to in on for with is are was were be been what which who when where why how did do does they their team".split())


def _keyword_terms(query: str, team_id: str) -> list[str]:
    """Return the literal normalized terms used by FTS and source admission."""

    words = re.findall(r"[^\W_]+", normalise_text(query).lower())
    team_word = "" if team_id == ALL_TEAMS else team_id.lower()
    return list(dict.fromkeys(word for word in words if word not in STOP_WORDS and word != team_word))


def _admission_terms(query: str, team_id: str) -> list[str]:
    """Canonicalize common inflections for stable turn-level prioritization."""

    canonical = []
    for term in _keyword_terms(query, team_id):
        if len(term) > 4 and term.endswith("ies"):
            term = term[:-3] + "y"
        elif len(term) > 4 and term.endswith("s") and not term.endswith("ss"):
            term = term[:-1]
        if term not in canonical:
            canonical.append(term)
    return canonical


def _keyword_expression(query: str, team_id: str) -> str:
    """Treat input as ordinary words, never as SQL or user-supplied FTS syntax."""

    return " OR ".join('"' + word + '"' for word in _keyword_terms(query, team_id))


def validate_search(query, team_id, as_of, limit):
    """Validate shared query filters before either retrieval strategy runs."""

    if not isinstance(query, str):
        raise ValueError("query must be text")
    if not isinstance(team_id, str) or not (re.fullmatch(r"[A-Z]{2}\d{4}", team_id.strip().upper())
                                            or team_id.strip().upper() == ALL_TEAMS):
        raise ValueError(f"team_id must be a team such as ES2002 or TS3012, or {ALL_TEAMS} for every team")
    team_id = team_id.strip().upper()
    # A team named by the user is a hard scope, not merely a ranking keyword.
    named_teams = set(re.findall(r"\b((?:ES|IS|TS)\d{4})(?:[a-d])?\b", query.upper(), re.IGNORECASE))
    if len(named_teams) == 1:
        requested_team = next(iter(named_teams))
        if team_id != ALL_TEAMS and team_id != requested_team:
            raise ValueError(f"Question names {requested_team}, but selected team is {team_id}. "
                             f"Select {requested_team} or All teams.")
        team_id = requested_team
    if not isinstance(as_of, str) or as_of not in ("a", "b", "c", "d"):
        raise ValueError("as_of must be a, b, c, or d")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    return team_id


def search_keywords(
    query: str,
    team_id: str,
    as_of: str = "d",
    limit: int = SEARCH_LIMIT,
    index_path: str | Path = BM25_INDEX_FILE,
    meeting_number: int | None = None,
) -> list[dict]:
    """Return BM25 matches; larger positive scores indicate greater relevance."""

    team_id = validate_search(query, team_id, as_of, limit)
    validate_meeting_filter(meeting_number, as_of)
    expression = _keyword_expression(query, team_id)
    if not expression:
        return []
    index_path = project_path(index_path).resolve()
    if not index_path.is_file():
        raise FileNotFoundError(f"Keyword index not found: {index_path}. Run scripts/build.py first.")
    connection = sqlite3.connect(index_path.as_uri() + "?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA user_version").fetchone()[0] != INDEX_VERSION:
            raise ValueError("Unsupported keyword index; rebuild with scripts/build.py")
        team_filter = "" if team_id == ALL_TEAMS else " AND chunks.team_id = ?"
        meeting_filter = " AND chunks.meeting_number = ?" if meeting_number is not None else ""
        parameters = [expression] + ([] if team_id == ALL_TEAMS else [team_id]) + ["abcd".index(as_of) + 1]
        if meeting_number is not None:
            parameters.append(meeting_number)
        parameters.append(limit)
        rows = connection.execute(
            "SELECT chunks.payload, bm25(chunks_fts) AS rank "
            "FROM chunks_fts JOIN chunks ON chunks.rowid = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ?" + team_filter + " AND chunks.meeting_number <= ? "
            + meeting_filter + " ORDER BY rank ASC, chunks.meeting_number ASC, chunks.chunk_id ASC LIMIT ?",
            parameters,
        ).fetchall()
    finally:
        connection.close()
    results = []
    query_terms = _admission_terms(query, team_id)
    for payload, rank in rows:
        result = json.loads(payload)
        result["score"] = -rank
        # Ephemeral retrieval metadata: used only to choose the most relevant
        # original turns inside a matched passage. It is not indexed evidence.
        result["_query_terms"] = query_terms
        results.append(result)
    return results


def validate_meeting_filter(meeting_number, as_of):
    """An exact meeting filter must also respect the caller's time cutoff."""

    if meeting_number is not None and (type(meeting_number) is not int or not 1 <= meeting_number <= "abcd".index(as_of) + 1):
        raise ValueError("meeting_number must be an integer within the meeting cutoff")


def fuse_results(keyword_results, vector_results):
    """Combine independent rankings without mixing incompatible raw scores."""

    combined = {}
    for method, results in (("bm25", keyword_results), ("embedding", vector_results)):
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            if chunk_id not in combined:
                combined[chunk_id] = dict(item, score=0.0, retrieval_scores={})
            combined[chunk_id]["score"] += 1 / (60 + rank)
            combined[chunk_id]["retrieval_scores"][method] = item["score"]
    results = list(combined.values())
    results.sort(key=lambda result: (-result["score"], result["meeting_number"], result["chunk_id"]))
    return results


def search_hybrid(query, team_id, as_of="d", limit=SEARCH_LIMIT, client=None,
                  keyword_index=BM25_INDEX_FILE, vector_index=VECTOR_INDEX_FILE, meeting_number=None):
    """Combine ranks, without mixing BM25 scores directly with cosine scores."""

    from app.vector_store import index_fingerprint, open_readonly, search_vectors
    from app.retrieval_plan import wants_current_decision, topic_terms, select_evidence
    from app.llm import NvidiaEmbeddingClient

    team_id = validate_search(query, team_id, as_of, limit)
    validate_meeting_filter(meeting_number, as_of)
    if not query.strip():
        return []
    vectors = open_readonly(vector_index)
    try:
        keywords = open_readonly(keyword_index)
        try:
            metadata = dict(vectors.execute("SELECT name, value FROM metadata"))
            if metadata.get("source_fingerprint") != index_fingerprint(keywords):
                raise ValueError("Chunks changed since embedding; rebuild the vector index")
            if team_id == ALL_TEAMS:
                latest = keywords.execute("SELECT MAX(meeting_number) FROM chunks WHERE meeting_number <= ?",
                                          ("abcd".index(as_of) + 1,)).fetchone()[0]
            else:
                latest = keywords.execute(
                    "SELECT MAX(meeting_number) FROM chunks WHERE team_id = ? AND meeting_number <= ?",
                    (team_id, "abcd".index(as_of) + 1),
                ).fetchone()[0]
        finally:
            keywords.close()
    finally:
        vectors.close()
    if latest is None:
        return []
    client = client or NvidiaEmbeddingClient()
    candidates = max(30, limit * 6)
    keyword_results = search_keywords(query, team_id, as_of, candidates, keyword_index, meeting_number=meeting_number)
    vector_results = search_vectors(query, team_id, as_of, candidates, client, vector_index, meeting_number=meeting_number)
    results = fuse_results(keyword_results, vector_results)
    terms = topic_terms(query, "" if team_id == ALL_TEAMS else team_id, STOP_WORDS)
    query_terms = sorted(terms)
    for result in results:
        result["_query_terms"] = query_terms
    followup = []
    # "Latest meeting" is per team, so the final-decision follow-up needs one team.
    if meeting_number is None and team_id != ALL_TEAMS and wants_current_decision(query) and terms:
        topic_query = " ".join(sorted(terms))
        latest_keywords = search_keywords(topic_query, team_id, as_of, candidates, keyword_index, meeting_number=latest)
        latest_vectors = search_vectors(query, team_id, as_of, candidates, client, vector_index, meeting_number=latest)
        followup = fuse_results(latest_keywords, latest_vectors)
        for result in followup:
            result["_query_terms"] = _admission_terms(topic_query, team_id)
    return select_evidence(results, limit, followup, terms)


def main() -> None:
    """Search from the project root with python -m app.retriever."""
    parser = argparse.ArgumentParser(description="Search meeting chunks using BM25, NVIDIA vectors, or both")
    parser.add_argument("query")
    parser.add_argument("--team", required=True)
    parser.add_argument("--as-of", choices=list("abcd"), default="d")
    parser.add_argument("--limit", type=int, default=SEARCH_LIMIT)
    parser.add_argument("--method", choices=("bm25", "embedding", "hybrid"), default="bm25")
    args = parser.parse_args()
    try:
        from app.vector_store import search_vectors
        from app.llm import EmbeddingError
        methods = {"bm25": search_keywords, "embedding": search_vectors, "hybrid": search_hybrid}
        results = methods[args.method](args.query, args.team, args.as_of, args.limit)
    except (ValueError, OSError, sqlite3.DatabaseError, EmbeddingError) as error:
        parser.exit(1, f"Search failed: {error}\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
