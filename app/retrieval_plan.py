"""Small, explicit retrieval rules for temporal coverage and repeated evidence."""

import re

from app.ingest import normalise_text

TEMPORAL_WORDS = set("final finally current currently latest last ultimately end ended choose chose chosen choice decide decided decision decisions settle settled design use uses used".split())
TOPIC_ALIASES = (
    {"power", "energy", "battery", "batteries", "kinetic", "solar", "wind", "crank", "rechargeable"},
    {"display", "screen", "lcd"},
    {"case", "casing", "housing", "material", "plastic", "rubber", "wood", "titanium"},
    {"cost", "price", "pricing", "budget", "total", "figure", "euro", "eur", "dollar", "pound"},
    {"remove", "removed", "drop", "dropped", "cut", "ditch", "replace", "replaced", "swap", "simplify", "take", "away"},
    {"television", "tv"},
    {"fruit", "fruits", "vegetable", "vegetables", "produce"},
    {"disagreement", "disagreements", "objection", "objections", "unresolved",
     "resolved", "resolve", "vote", "voted"},
)


def requested_meeting(question: str) -> int | None:
    """Return one explicitly requested phase, leaving temporal comparisons open."""

    if re.search(r"\b(before|after|between|across|through|until|since|compare|compared|comparison|evolve|evolved|change|changed)\b|\bas of\b|\bup to\b", question, re.I):
        return None
    patterns = (
        r"\bkick[ -]?off\b",
        r"\bfunctional(?:\s+design)?\s+meeting\b|\bfunctional\s+design\b",
        r"\bconceptual(?:\s+design)?\s+meeting\b|\bconceptual\s+design\b",
        r"\bdetailed(?:\s+design)?\s+meeting\b|\bdetailed\s+design\b",
    )
    numbers = {number for number, pattern in enumerate(patterns, 1)
               if re.search(pattern, question, re.I)}
    for match in re.finditer(r"\b(?:ES|IS|TS)\d{4}([a-d])\b|\bmeeting\s+([a-d])\b", question, re.I):
        numbers.add("abcd".index((match[1] or match[2]).lower()) + 1)
    return next(iter(numbers)) if len(numbers) == 1 else None


def wants_current_decision(query):
    """Detect an explicit current/final-state request, not ordinary topic search."""

    return bool(re.search(r"\b(final|finally|current|currently|latest|ultimately)\b|\bin the end\b", query, re.I))


def topic_terms(query, team_id, stop_words):
    """Expand topic vocabulary without adding an answer or using a model call."""

    terms = set(re.findall(r"[^\W_]+", normalise_text(query).lower()))
    terms -= stop_words | TEMPORAL_WORDS | {team_id.lower()}
    for aliases in TOPIC_ALIASES:
        if terms & aliases:
            terms.update(aliases)
    return terms


def decision_evidence_score(chunk, terms):
    """Prefer topical decision language over tentative recaps within a passage.

    This is a retrieval heuristic, not proof that a decision is current. Evaluate
    the returned turns with a reasoning agent before writing a final answer.
    """

    turns = re.split(r"(?m)^\d{1,2}:\d{2}:\d{2}\s+Speaker\s+\w+\s*\n", chunk["original_text"])
    best = -100
    for turn in turns:
        words = set(re.findall(r"[^\W_]+", normalise_text(turn).lower()))
        matches = len(words & terms)
        if not matches:
            continue
        score = min(matches, 3)
        if re.search(r"\b(decided|agreed|chosen|settled|instead|only|just)\b", turn, re.I):
            score += 3
        if re.search(r"\b(no|not|dropped|removed|rejected)\b", turn, re.I):
            score += 1
        if re.search(r"\b(recap|hopefully|proposed|proposal|considering)\b|\blast time\b", turn, re.I):
            score -= 5
        best = max(best, score)
    return best


def overlaps_evidence(left, right, threshold=0.35):
    """Detect substantially shared source spans, without merging speaker turns."""

    if left["meeting_id"] != right["meeting_id"]:
        return False
    if left["chunk_id"] == right["chunk_id"]:
        return True
    if all("start_char" in item and "end_char" in item for item in (left, right)):
        shared = max(0, min(left["end_char"], right["end_char"]) - max(left["start_char"], right["start_char"]))
        length = min(left["end_char"] - left["start_char"], right["end_char"] - right["start_char"])
        if length > 0:
            return shared / length >= threshold
    left_turns, right_turns = set(left["turn_ids"]), set(right["turn_ids"])
    shortest = min(len(left_turns), len(right_turns))
    return shortest > 0 and len(left_turns & right_turns) / shortest >= 0.5


def select_evidence(ranked, limit, followup=None, terms=None):
    """Reserve up to two topical latest-meeting slots, then keep diverse evidence."""

    selected = []
    followup = followup or []
    if followup and terms:
        ordered = sorted(followup, key=lambda item: (-decision_evidence_score(item, terms), -item["score"], item["chunk_id"]))
        for item in ordered:
            if decision_evidence_score(item, terms) < 1:
                continue
            if not any(overlaps_evidence(item, kept) for kept in selected):
                selected.append(dict(item, retrieval_reason="Latest available meeting: targeted decision search"))
            if len(selected) >= min(2, limit):
                break
    for item in ranked:
        if len(selected) >= limit:
            break
        if not any(overlaps_evidence(item, kept) for kept in selected):
            selected.append(dict(item, retrieval_reason="Hybrid relevance with overlap reduction"))
    return selected
