"""Verify quotations using only case and whitespace normalisation."""

import re


def _comparable(text: str) -> str:
    return " ".join(text.lower().split())


def verify_quote(quote: str, parsed_meeting: dict) -> bool:
    """Accept at least six words found contiguously within one original utterance."""
    if not isinstance(quote, str) or len(quote.split()) < 6:
        return False
    candidate = _comparable(quote)
    pattern = re.compile(r"(?<!\w)" + re.escape(candidate) + r"(?!\w)")
    return any(pattern.search(_comparable(turn["original_text"])) for turn in parsed_meeting["utterances"])


def verify_citation(citation: dict, meetings: dict[str, dict], team_id: str | None = None, as_of: str = "d") -> bool:
    """Check meeting identity, team, time cutoff and quote before output."""
    meeting_id = citation.get("meeting_id")
    if not isinstance(meeting_id, str) or meeting_id not in meetings:
        return False
    meeting = meetings[meeting_id]
    if team_id is not None and meeting["team_id"] != team_id:
        return False
    if as_of not in "abcd" or len(as_of) != 1:
        raise ValueError("as_of must be a, b, c, or d")
    if meeting.get("meeting_sequence") not in list("abcd"[:"abcd".index(as_of) + 1]):
        return False
    return verify_quote(citation.get("quote"), meeting)


def quote_from_turn(turn: dict, max_words: int = 40) -> str:
    """Copy a word-aligned source span without changing punctuation or spacing."""
    if max_words < 6:
        raise ValueError("max_words must be at least six")
    text = turn["original_text"]
    words = list(re.finditer(r"\S+", text))
    if len(words) < 6:
        raise ValueError("Turn is too short for a citation")
    return text[words[0].start():words[min(max_words, len(words)) - 1].end()]


def quote_with_context(quote: str, turn: dict, max_words: int = 40) -> str:
    """Copy context around a unique exact source span, without rewriting words."""

    if type(max_words) is not int or max_words < 6:
        raise ValueError("max_words must be an integer of at least six")
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("Quote must be nonempty text")
    text = turn["original_text"]
    words = list(re.finditer(r"\S+", text))
    if len(words) < 6:
        raise ValueError("The supplied turn is too short for a six-word citation")
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(word) for word in quote.split()) + r"(?!\w)"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))
    if len(matches) != 1:
        raise ValueError("Short quote must match one unique exact span in the supplied turn")
    match = matches[0]
    first = next(index for index, word in enumerate(words) if word.end() > match.start())
    last = next((index for index, word in enumerate(words) if word.start() >= match.end()), len(words))
    if last - first > max_words:
        raise ValueError("The selected span exceeds the quote word budget")
    start = max(0, first - (max_words - (last - first)) // 2)
    end = min(len(words), start + max_words)
    start = max(0, end - max_words)
    result = text[words[start].start():words[end - 1].end()]
    if not verify_quote(result, {"utterances": [turn]}):
        raise ValueError("Expanded quote failed original-turn verification")
    return result
