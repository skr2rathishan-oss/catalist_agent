"""Incremental, source-checked meeting memory stored only under data/processed."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re

from app.config import MEMORY_CONTEXT_MAX_WORDS, MEMORY_DIR, PARSED_DIR, project_path
from app.citations import verify_quote, quote_with_context
from app.generation import GenerationError
from app.memory import (build_memory_prompt, memory_as_of, record_to_dict, source_turns,
                        topic_terms, validate_extraction, validate_ledger, validate_record)
from app.agents.prompts import MEMORY_PROMPT_VERSION, MEMORY_RECONCILIATION_INSTRUCTIONS


def _write_json(path: Path, value: dict) -> None:
    """Replace one derived file after its complete contents are ready."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_hash(meeting: dict) -> str:
    return hashlib.sha256(meeting["original_transcript_text"].encode("utf-8")).hexdigest()


def turn_windows(meeting: dict, max_words: int = MEMORY_CONTEXT_MAX_WORDS) -> list[list[int]]:
    """Cover every original turn exactly once within the extraction word budget."""

    if type(max_words) is not int or not 1 <= max_words <= MEMORY_CONTEXT_MAX_WORDS:
        raise ValueError("Invalid memory window size")
    windows, current, words = [], [], 0
    for turn in meeting["utterances"]:
        size = len(turn["original_text"].split())
        if size > max_words:
            raise ValueError(f"Turn {turn['utterance_id']} exceeds the memory word budget")
        if current and words + size > max_words:
            windows.append(current)
            current, words = [], 0
        current.append(turn["utterance_id"])
        words += size
    if current:
        windows.append(current)
    return windows


def _copy_short_quotes(response: dict, meeting: dict, turn_ids: list[int]) -> dict:
    """Copy a relevant exact span from the cited original turn when needed."""

    if not isinstance(response, dict) or not isinstance(response.get("records"), list):
        return response
    turns = source_turns(meeting, turn_ids)
    copied = json.loads(json.dumps(response))
    for record in copied["records"]:
        if not isinstance(record, dict) or not isinstance(record.get("evidence"), list):
            continue
        for evidence in record["evidence"]:
            if not isinstance(evidence, dict):
                continue
            turn = turns.get(evidence.get("turn_id"))
            quote = evidence.get("quote")
            if turn and isinstance(quote, str) and not verify_quote(quote, {"utterances": [turn]}):
                try:
                    evidence["quote"] = quote_with_context(quote, turn)
                except ValueError:
                    replacement = _source_excerpt(turn["original_text"],
                                                  record.get("summary", "") + " " + quote)
                    if replacement is not None:
                        evidence["quote"] = replacement
    return copied


def _source_excerpt(text: str, hint: str, max_words: int = 40) -> str | None:
    """Choose an exact source window with lexical support for the model's claim."""

    words = list(re.finditer(r"\S+", text))
    if len(words) < 6:
        return None
    significant = lambda value: {token for token in re.findall(r"[a-z0-9]+", value.casefold())
                                  if len(token) > 3}
    targets = significant(hint)
    if not targets:
        return None
    width = min(max_words, len(words))
    scored = []
    for start in range(len(words) - width + 1):
        excerpt = text[words[start].start():words[start + width - 1].end()]
        scored.append((len(significant(excerpt) & targets), start, excerpt))
    overlap, _, result = max(scored, key=lambda item: (item[0], -item[1]))
    return result if overlap >= min(2, len(targets)) else None


def _normalise_source_fields(item: dict, meeting: dict, turn_ids: list[int], number: int) -> dict:
    """Supply identities and unambiguous extraction-contract defaults only."""

    if not isinstance(item, dict):
        return item
    result = dict(item)
    result["event_id"] = f"{meeting['meeting_id']}_draft_{number:03d}"
    result.setdefault("meeting_id", meeting["meeting_id"])
    result.setdefault("team_id", meeting["team_id"])
    result.setdefault("supersedes", [])
    result.setdefault("owner", None)
    if result["owner"] is None:
        result.setdefault("owner_source_turn_ids", [])
    default_status = {"decision": "agreed", "change": "agreed", "number": "recorded"}
    if result.get("type") in default_status:
        result.setdefault("status", default_status[result["type"]])
    if isinstance(result.get("source_turn_ids"), list):
        try:
            speakers = {turn["speaker_id"] for turn in
                        source_turns(meeting, result["source_turn_ids"]).values()}
            if len(speakers) == 1:
                result["speaker_id"] = speakers.pop()
        except ValueError:
            pass
    return result


def _extract_window(meeting: dict, turn_ids: list[int], client) -> tuple[list, int]:
    """Validate every event in one model response; keep source-valid siblings."""

    prompt = build_memory_prompt(meeting, turn_ids)
    payload = {"prompt_version": prompt["prompt_version"],
               "source_context": json.loads(prompt["source_context"])}
    response = _copy_short_quotes(client.generate_json(prompt["system"], payload), meeting, turn_ids)
    if not isinstance(response, dict) or set(response) != {"records"} or not isinstance(response["records"], list):
        raise ValueError("Memory extraction has invalid JSON fields")
    if len(response["records"]) > len(turn_ids) * 5:
        raise ValueError("Memory extraction returned too many records")
    extracted, invalid = [], []
    for index, original in enumerate(response["records"], 1):
        item = _normalise_source_fields(original, meeting, turn_ids, index)
        try:
            extracted.extend(validate_extraction({"records": [item]}, meeting, turn_ids))
        except (ValueError, TypeError, KeyError) as error:
            invalid.append({"record": item, "error": str(error)})
    repaired = 0
    if invalid:
        try:
            corrected = _copy_short_quotes(client.generate_json(
                prompt["system"] + "\nCorrect only invalid_records. Return their corrected records or an empty list if unsupported.",
                dict(payload, invalid_records=invalid)), meeting, turn_ids)
        except (ValueError, RuntimeError):
            corrected = None
        if isinstance(corrected, dict) and set(corrected) == {"records"} and isinstance(corrected["records"], list):
            for index, original in enumerate(corrected["records"], len(response["records"]) + 1):
                item = _normalise_source_fields(original, meeting, turn_ids, index)
                try:
                    extracted.extend(validate_extraction({"records": [item]}, meeting, turn_ids))
                    repaired += 1
                except (ValueError, TypeError, KeyError):
                    pass
    if response["records"] and not extracted:
        raise ValueError(f"No valid memory records in {meeting['meeting_id']}")
    return extracted, max(0, len(invalid) - repaired)


def _structural_failure(error: Exception) -> bool:
    return isinstance(error, ValueError) or (isinstance(error, GenerationError)
                                               and error.status == "invalid_response")


def _split_turn_ids(meeting: dict, turn_ids: list[int]) -> tuple[list[int], list[int]]:
    sizes = {turn["utterance_id"]: len(turn["original_text"].split())
             for turn in meeting["utterances"] if turn["utterance_id"] in turn_ids}
    half = sum(sizes[turn_id] for turn_id in turn_ids) / 2
    total, split_at = 0, 1
    for index, turn_id in enumerate(turn_ids[:-1], 1):
        total += sizes[turn_id]
        split_at = index
        if total >= half:
            break
    return turn_ids[:split_at], turn_ids[split_at:]


def _extract_resilient(meeting: dict, turn_ids: list[int], client, fallback_client=None):
    """Subdivide empty or malformed long windows; use fallback on smaller ones."""

    word_count = sum(len(turn["original_text"].split()) for turn in meeting["utterances"]
                     if turn["utterance_id"] in turn_ids)
    long_window = word_count > 900 and len(turn_ids) > 1
    try:
        try:
            extracted, rejected = _extract_window(meeting, turn_ids, client)
        except (ValueError, GenerationError) as error:
            if not _structural_failure(error) or long_window or fallback_client is None:
                raise
            extracted, rejected = _extract_window(meeting, turn_ids, fallback_client)
            if not extracted and len(turn_ids) > 1:
                raise ValueError("Both models returned no usable records for a long window")
            return extracted, rejected, 1, 1
        if not extracted and long_window:
            raise ValueError("No records from a long memory window")
        if fallback_client is not None and not long_window and (
                not extracted or rejected > max(2, len(extracted) // 2)):
            try:
                alternative, alternative_rejected = _extract_window(
                    meeting, turn_ids, fallback_client)
                if (len(alternative) > len(extracted)
                        or (len(alternative) == len(extracted)
                            and alternative_rejected < rejected)):
                    return alternative, alternative_rejected, 1, 1
            except (ValueError, RuntimeError) as error:
                if not extracted and _structural_failure(error):
                    raise
        return extracted, rejected, 0, 1
    except (ValueError, GenerationError) as error:
        if not _structural_failure(error) or len(turn_ids) < 2:
            raise
        left, right = _split_turn_ids(meeting, turn_ids)
        first = _extract_resilient(meeting, left, client, fallback_client)
        second = _extract_resilient(meeting, right, client, fallback_client)
        return (first[0] + second[0], first[1] + second[1],
                first[2] + second[2], first[3] + second[3])


def extract_meeting(meeting: dict, client, memory_dir=MEMORY_DIR,
                    max_words: int = MEMORY_CONTEXT_MAX_WORDS, fallback_client=None,
                    retry_rejected: bool = False) -> dict:
    """Extract bounded windows, validate quotes, and reuse matching local results."""

    memory_dir = project_path(memory_dir)
    path = memory_dir / f"{meeting['meeting_id']}.json"
    model = client.settings.model + (f"+{fallback_client.settings.model}" if fallback_client else "")
    max_tokens = getattr(client.settings, "max_tokens", None)
    digest = _source_hash(meeting)
    saved_for_retry = None
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if (existing.get("source_sha256") == digest and existing.get("model") == model
                and existing.get("prompt_version") == MEMORY_PROMPT_VERSION
                and existing.get("generation_max_tokens") == max_tokens
                and existing.get("max_words") == max_words
                and isinstance(existing.get("records"), list)):
            for record in existing["records"]:
                validate_record(record, {meeting["meeting_id"]: meeting})
            if not retry_rejected or existing.get("rejected_records", 0) == 0:
                return {"meeting_id": meeting["meeting_id"], "records": len(existing["records"]),
                        "windows": 0, "rejected_records": existing.get("rejected_records", 0),
                        "fallback_windows": existing.get("fallback_windows", 0), "cached": True}
            saved_for_retry = existing
    records, rejected_records, fallback_windows, actual_windows = [], 0, 0, 0
    windows = turn_windows(meeting, max_words)
    for number, turn_ids in enumerate(windows, 1):
        extracted, rejected, fallbacks, leaves = _extract_resilient(
            meeting, turn_ids, client, fallback_client)
        fallback_windows += fallbacks
        actual_windows += leaves
        rejected_records += rejected
        for index, record in enumerate(extracted, 1):
            record = replace(record, event_id=f"{meeting['meeting_id']}_w{number:03d}_e{index:03d}")
            records.append(record_to_dict(record))
    if saved_for_retry is not None and (
            len(records), -rejected_records) <= (
                len(saved_for_retry["records"]), -saved_for_retry["rejected_records"]):
        return {"meeting_id": meeting["meeting_id"],
                "records": len(saved_for_retry["records"]), "windows": actual_windows,
                "rejected_records": saved_for_retry["rejected_records"],
                "fallback_windows": saved_for_retry.get("fallback_windows", 0),
                "cached": True, "retained_existing": True}
    _write_json(path, {"meeting_id": meeting["meeting_id"], "source_sha256": digest,
                       "model": model, "prompt_version": MEMORY_PROMPT_VERSION,
                       "max_words": max_words, "generation_max_tokens": max_tokens,
                       "rejected_records": rejected_records, "fallback_windows": fallback_windows,
                       "actual_windows": actual_windows,
                       "records": records})
    return {"meeting_id": meeting["meeting_id"], "records": len(records),
            "windows": actual_windows, "rejected_records": rejected_records,
            "fallback_windows": fallback_windows, "cached": False}


def _read_team(team_id: str, parsed_dir: Path, memory_dir: Path):
    meetings, records = {}, []
    for phase in "abcd":
        meeting_id = team_id + phase
        source = parsed_dir / f"{meeting_id}.json"
        saved = memory_dir / f"{meeting_id}.json"
        if not saved.is_file():
            continue
        if not source.is_file():
            raise ValueError(f"Missing parsed source for memory: {meeting_id}")
        meeting = json.loads(source.read_text(encoding="utf-8"))
        data = json.loads(saved.read_text(encoding="utf-8"))
        if data["source_sha256"] != _source_hash(meeting):
            raise ValueError(f"Stale memory for {meeting_id}; rebuild it")
        meetings[meeting_id] = meeting
        records.extend(data["records"])
    return meetings, records


def _family(record: dict) -> str | None:
    return {"decision": "decision", "proposal": "decision", "change": "decision",
            "action_item": "action", "objection": "disagreement",
            "disagreement": "disagreement"}.get(record["type"])


def _replacement_candidates(newer: dict, records: list[dict]) -> list[dict]:
    """Bound model checks to earlier, plausibly related events in one team."""

    terms = topic_terms(newer["topic"])
    candidates = []
    for older in records:
        if (older["meeting_id"] >= newer["meeting_id"]
                or older["team_id"] != newer["team_id"]
                or _family(older) != _family(newer)):
            continue
        exact = older["topic"].strip().casefold() == newer["topic"].strip().casefold()
        overlap = len(terms & topic_terms(older["topic"]))
        if exact or overlap:
            candidates.append((exact, overlap, older["meeting_id"], older))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [item[3] for item in candidates[:4]]


def reconcile_team(team_id: str, client, parsed_dir=PARSED_DIR, memory_dir=MEMORY_DIR) -> dict:
    """Ask about plausible same-topic replacements; validate every proposed link."""

    parsed_dir, memory_dir = project_path(parsed_dir), project_path(memory_dir)
    meetings, records = _read_team(team_id, parsed_dir, memory_dir)
    if not records:
        return {"team_id": team_id, "records": 0, "links": 0, "cached": True}
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()
    path = memory_dir / "links" / f"{team_id}.json"
    model = client.settings.model
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("source_sha256") == fingerprint and existing.get("model") == model:
            linked = _apply_links(records, existing["links"])
            validate_ledger(linked, meetings)
            return {"team_id": team_id, "records": len(records),
                    "links": len(existing["links"]), "cached": True}
    links = []
    for newer in records:
        family = _family(newer)
        if family is None or newer["type"] == "proposal":
            continue
        candidates = _replacement_candidates(newer, records)
        if not candidates:
            continue
        response = client.generate_json(MEMORY_RECONCILIATION_INSTRUCTIONS, {
            "new_event": _event_summary(newer),
            "earlier_events": [_event_summary(item) for item in candidates],
        })
        if not isinstance(response, dict) or set(response) != {"supersedes"} or not isinstance(response["supersedes"], list):
            raise ValueError("Reconciliation must return a supersedes list")
        allowed = {item["event_id"] for item in candidates}
        selected = response["supersedes"]
        if len(selected) != len(set(selected)) or any(item not in allowed for item in selected):
            raise ValueError("Reconciliation selected an unknown or duplicate event")
        links.extend({"new_event_id": newer["event_id"], "old_event_id": item} for item in selected)
    validate_ledger(_apply_links(records, links), meetings)
    _write_json(path, {"team_id": team_id, "source_sha256": fingerprint,
                       "model": model, "links": links})
    return {"team_id": team_id, "records": len(records), "links": len(links), "cached": False}


def _event_summary(record: dict) -> dict:
    return {key: record[key] for key in ("event_id", "type", "meeting_id", "topic", "summary",
                                         "status", "evidence")}


def _apply_links(records: list[dict], links: list[dict]) -> list[dict]:
    by_id = {item["event_id"]: dict(item, supersedes=list(item["supersedes"])) for item in records}
    for link in links:
        if not isinstance(link, dict) or set(link) != {"new_event_id", "old_event_id"}:
            raise ValueError("Invalid memory link")
        if link["new_event_id"] not in by_id:
            raise ValueError("Memory link has unknown new event")
        by_id[link["new_event_id"]]["supersedes"].append(link["old_event_id"])
    return list(by_id.values())


def load_team_memory(team_id: str, as_of="d", parsed_dir=PARSED_DIR,
                     memory_dir=MEMORY_DIR) -> list:
    """Return a validated, cutoff-aware snapshot from local derived files."""

    parsed_dir, memory_dir = project_path(parsed_dir), project_path(memory_dir)
    meetings, records = _read_team(team_id, parsed_dir, memory_dir)
    if not records:
        return []
    path = memory_dir / "links" / f"{team_id}.json"
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()
        if data["source_sha256"] != fingerprint:
            raise ValueError(f"Stale memory links for {team_id}; reconcile again")
        records = _apply_links(records, data["links"])
    return memory_as_of(records, meetings, team_id, as_of)


def memory_hints(question: str, team_id: str, as_of="d", parsed_dir=PARSED_DIR,
                 memory_dir=MEMORY_DIR, limit=8) -> list[dict]:
    """Select small planning hints; transcript retrieval remains the evidence source."""

    records = load_team_memory(team_id, as_of, parsed_dir, memory_dir)
    stopwords = {"about", "and", "are", "did", "does", "finally", "for", "from",
                 "how", "into", "meeting", "team", "the", "their", "them", "was",
                 "were", "what", "when", "where", "which", "who", "why"}
    terms = {term for term in re.findall(r"[a-z0-9]+", question.casefold())
             if len(term) > 2 and term not in stopwords}
    ranked = sorted(records, key=lambda item: (
        -len(terms & set(re.findall(r"[a-z0-9]+",
                                    (item.topic + " " + item.summary).casefold()))),
        item.status == "superseded", -item.meeting_number, item.event_id))
    return [{"meeting_id": item.meeting_id, "topic": item.topic,
             "summary": item.summary, "status": item.status,
             "source_turn_ids": list(item.source_turn_ids)}
            for item in ranked[:limit]
            if terms & set(re.findall(r"[a-z0-9]+",
                                      (item.topic + " " + item.summary).casefold()))]
