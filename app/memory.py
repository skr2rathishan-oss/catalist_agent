"""Structured meeting memory contracts and deterministic source validation.

This module does not call a model or judge whether a summary follows logically
from a quote. It validates the model's proposed structure and source references.
"""

from dataclasses import asdict, dataclass, replace
import json
import re

from app.agents.prompts import MEMORY_EXTRACTION_INSTRUCTIONS, MEMORY_PROMPT_VERSION
from app.citations import verify_quote
from app.config import MEMORY_CONTEXT_MAX_WORDS

STATUSES = {
    "decision": {"agreed", "superseded"},
    "proposal": {"proposed", "rejected", "superseded"},
    "action_item": {"open", "completed", "cancelled", "superseded"},
    "objection": {"open", "resolved", "superseded"},
    "disagreement": {"open", "resolved", "superseded"},
    "change": {"agreed", "superseded"},
    "number": {"recorded"},
}
FIELDS = {"event_id", "type", "team_id", "meeting_id", "speaker_id", "topic", "summary",
          "source_turn_ids", "evidence", "status", "owner", "owner_source_turn_ids", "supersedes"}


@dataclass(frozen=True)
class Evidence:
    turn_id: int
    quote: str


@dataclass(frozen=True)
class MemoryRecord:
    event_id: str
    type: str
    team_id: str
    meeting_id: str
    meeting_number: int
    speaker_id: str
    topic: str
    summary: str
    source_turn_ids: tuple[int, ...]
    evidence: tuple[Evidence, ...]
    status: str
    owner: str | None
    owner_source_turn_ids: tuple[int, ...]
    supersedes: tuple[str, ...]


def source_turns(meeting, turn_ids):
    """Resolve source IDs and verify their original-text character spans."""

    if not isinstance(turn_ids, list) or not turn_ids or any(type(value) is not int or value < 1 for value in turn_ids):
        raise ValueError("Source turn IDs must be a nonempty list of positive integers")
    if len(set(turn_ids)) != len(turn_ids):
        raise ValueError("Duplicate source turn IDs")
    available = {turn["utterance_id"]: turn for turn in meeting["utterances"]}
    selected = {}
    for turn_id in turn_ids:
        if turn_id not in available:
            raise ValueError(f"Unknown source turn: {turn_id}")
        turn = available[turn_id]
        start, end = turn["start_char"], turn["end_char"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(meeting["original_transcript_text"]):
            raise ValueError(f"Invalid source offsets: {turn_id}")
        if meeting["original_transcript_text"][turn["start_char"]:turn["end_char"]] != turn["original_text"]:
            raise ValueError(f"Source text/offset mismatch: {turn_id}")
        selected[turn_id] = turn
    return selected


def validate_record(data, meetings, allowed_turn_ids=None, as_of="d"):
    """Return an immutable record only after validating its evidence boundary."""

    if not isinstance(data, dict) or set(data) != FIELDS:
        raise ValueError("Memory record has missing or unknown fields")
    for name in ("event_id", "type", "team_id", "meeting_id", "speaker_id", "topic", "summary", "status"):
        if not isinstance(data[name], str) or not data[name].strip():
            raise ValueError(f"Memory field must be nonempty text: {name}")
    meeting_id = data["meeting_id"]
    if not re.fullmatch(r"[A-Z]{2}\d{4}[a-d]", meeting_id) or meeting_id not in meetings:
        raise ValueError("Unknown memory meeting")
    if as_of not in ("a", "b", "c", "d") or meeting_id[-1] > as_of:
        raise ValueError("Memory meeting exceeds cutoff")
    meeting = meetings[meeting_id]
    if meeting["meeting_id"] != meeting_id or data["team_id"] != meeting["team_id"] or data["team_id"] != meeting_id[:-1]:
        raise ValueError("Memory team does not match its source meeting")
    number = "abcd".index(meeting_id[-1]) + 1
    if meeting["meeting_number"] != number or meeting["meeting_sequence"] != meeting_id[-1]:
        raise ValueError("Source meeting order is inconsistent")
    if not re.fullmatch(re.escape(meeting_id) + r"_[A-Za-z0-9_-]+", data["event_id"]):
        raise ValueError("Event ID must start with its meeting ID and underscore")
    if data["type"] not in STATUSES or data["status"] not in STATUSES[data["type"]]:
        raise ValueError("Invalid memory type/status combination")
    turns = source_turns(meeting, data["source_turn_ids"])
    if allowed_turn_ids is not None and not set(turns).issubset(allowed_turn_ids):
        raise ValueError("Record cites a turn outside the supplied context")
    if data["speaker_id"] not in {turn["speaker_id"] for turn in turns.values()}:
        raise ValueError("Attributed speaker does not appear in the source turns")
    if not isinstance(data["evidence"], list) or not data["evidence"]:
        raise ValueError("Every memory record needs exact evidence")
    evidence = []
    seen = set()
    for item in data["evidence"]:
        if not isinstance(item, dict) or set(item) != {"turn_id", "quote"} or type(item["turn_id"]) is not int:
            raise ValueError("Invalid memory evidence format")
        turn_id = item["turn_id"]
        if turn_id not in turns or not verify_quote(item["quote"], {"utterances": [turns[turn_id]]}):
            raise ValueError("Quote must match one referenced original turn and have six words")
        key = (turn_id, " ".join(item["quote"].lower().split()))
        if key in seen:
            raise ValueError("Duplicate memory evidence")
        seen.add(key)
        evidence.append(Evidence(turn_id, item["quote"]))
    owner = data["owner"]
    owner_ids = data["owner_source_turn_ids"]
    if not isinstance(owner_ids, list) or any(type(value) is not int for value in owner_ids) or len(owner_ids) != len(set(owner_ids)):
        raise ValueError("Invalid owner source turn IDs")
    if data["type"] != "action_item" and (owner is not None or owner_ids):
        raise ValueError("Only action items have owners")
    if owner is None and owner_ids:
        raise ValueError("Unknown owner cannot have assignment source IDs")
    if owner is not None:
        speakers = {turn["speaker_id"] for turn in meeting["utterances"]}
        if not isinstance(owner, str) or owner not in speakers or not owner_ids or not set(owner_ids).issubset(turns):
            raise ValueError("Action owner requires a known speaker and assignment evidence")
        if not set(owner_ids).intersection(item.turn_id for item in evidence):
            raise ValueError("Owner assignment requires an exact evidence quote")
    links = data["supersedes"]
    if not isinstance(links, list) or any(not isinstance(link, str) or not link for link in links) or len(links) != len(set(links)):
        raise ValueError("Invalid supersession links")
    return MemoryRecord(data["event_id"], data["type"], data["team_id"], meeting_id, number,
                        data["speaker_id"], data["topic"], data["summary"], tuple(turns), tuple(evidence),
                        data["status"], owner, tuple(owner_ids), tuple(links))


def record_to_dict(record):
    """Return the extraction contract; meeting order is always derived on load."""

    result = asdict(record)
    result.pop("meeting_number")
    return json.loads(json.dumps(result))


def validate_ledger(records, meetings):
    """Validate explicit links; never infer supersession from meeting recency."""

    if not isinstance(records, list):
        raise ValueError("Memory ledger must be a list of record objects")
    validated = [validate_record(record, meetings) for record in records]
    by_id = {record.event_id: record for record in validated}
    if len(by_id) != len(validated):
        raise ValueError("Duplicate memory event IDs")
    targeted = set()
    families = {"decision": "decision", "proposal": "decision", "change": "decision",
                "action_item": "action", "objection": "disagreement", "disagreement": "disagreement"}
    for newer in validated:
        for old_id in newer.supersedes:
            if old_id not in by_id:
                raise ValueError("Supersession target does not exist")
            older = by_id[old_id]
            if newer.team_id != older.team_id or " ".join(newer.topic.lower().split()) != " ".join(older.topic.lower().split()):
                raise ValueError("Supersession requires the same team and topic")
            if newer.type not in families or families[newer.type] != families.get(older.type):
                raise ValueError("Incompatible supersession event types")
            if (newer.meeting_number, min(newer.source_turn_ids)) <= (older.meeting_number, max(older.source_turn_ids)):
                raise ValueError("Supersession must point to strictly earlier evidence")
            if newer.type == "proposal":
                raise ValueError("A tentative proposal cannot supersede an earlier event")
            targeted.add(old_id)
    for record in validated:
        if record.status == "superseded" and record.event_id not in targeted:
            raise ValueError("Superseded status needs a checked replacement link")
    return sorted(validated, key=lambda record: (record.team_id, record.meeting_number, min(record.source_turn_ids), record.event_id))


def memory_as_of(records, meetings, team_id, as_of="d"):
    """Derive a timeline using only visible replacement links; preserve inputs."""

    if not re.fullmatch(r"[A-Z]{2}\d{4}", team_id) or as_of not in ("a", "b", "c", "d"):
        raise ValueError("Invalid memory team or cutoff")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise ValueError("Timeline input must be original record objects")
    if any(record.get("status") == "superseded" for record in records):
        raise ValueError("Use original event statuses; superseded is a derived timeline status")
    visible = [record for record in validate_ledger(records, meetings) if record.team_id == team_id and record.meeting_id[-1] <= as_of]
    replaced = {old_id for record in visible for old_id in record.supersedes}
    return [replace(record, status="superseded") if record.event_id in replaced else record for record in visible]


def validate_extraction(response, meeting, turn_ids):
    """Check strict JSON extraction against only the passage the model received."""

    selected = source_turns(meeting, turn_ids)
    data = json.loads(response) if isinstance(response, str) else response
    if not isinstance(data, dict) or set(data) != {"records"} or not isinstance(data["records"], list):
        raise ValueError("Extraction must be a JSON object containing a records list")
    if len(data["records"]) > len(selected) * 5:
        raise ValueError("Too many extracted records for the supplied context")
    records = []
    for item in data["records"]:
        record = validate_record(item, {meeting["meeting_id"]: meeting}, set(selected))
        if record.supersedes or record.status == "superseded":
            raise ValueError("Passage extraction cannot reconcile unseen prior memory")
        records.append(record)
    if len({record.event_id for record in records}) != len(records):
        raise ValueError("Duplicate extracted event IDs")
    return records


def build_memory_prompt(meeting, turn_ids, max_words=MEMORY_CONTEXT_MAX_WORDS):
    """Build bounded extraction context from explicitly selected original turns."""

    selected = source_turns(meeting, turn_ids)
    if type(max_words) is not int or max_words < 1 or max_words > MEMORY_CONTEXT_MAX_WORDS:
        raise ValueError("Invalid memory context word budget")
    if sum(len(turn["original_text"].split()) for turn in selected.values()) > max_words:
        raise ValueError("Memory context is too large; select fewer source turns")
    context = {"team_id": meeting["team_id"], "meeting_id": meeting["meeting_id"],
               "meeting_number": meeting["meeting_number"],
               "turns": [{"turn_id": turn_id, "speaker_id": turn["speaker_id"],
                          "timestamp": turn["timestamp"], "original_text": turn["original_text"]}
                         for turn_id, turn in sorted(selected.items())]}
    return {"prompt_version": MEMORY_PROMPT_VERSION, "system": MEMORY_EXTRACTION_INSTRUCTIONS,
            "source_context": json.dumps(context, ensure_ascii=False)}
