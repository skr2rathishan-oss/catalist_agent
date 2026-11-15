"""Deterministic ingestion and conservative normalisation for search only."""

import json
import re
from pathlib import Path

from app.config import CHUNKS_DIR, PARSED_DIR, RAW_DIR, project_path

SMALL_NUMBERS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split(),
    range(20),
))
TENS = dict(zip("twenty thirty forty fifty sixty seventy eighty ninety".split(), range(20, 100, 10)))
NUMBER_WORDS = {**SMALL_NUMBERS, **TENS}
NUMBER_PATTERN = re.compile(
    r"\b(?:" + "|".join(NUMBER_WORDS) + r")(?:[ -]+(?:" + "|".join(NUMBER_WORDS) + r"))*\b",
    re.IGNORECASE,
)


def _number(match: re.Match) -> str:
    words = re.split(r"[ -]+", match.group().lower())
    if len(words) == 1:
        return str(NUMBER_WORDS[words[0]])
    if len(words) == 2 and words[0] in TENS and words[1] in SMALL_NUMBERS and 0 < SMALL_NUMBERS[words[1]] < 10:
        return str(NUMBER_WORDS[words[0]] + NUMBER_WORDS[words[1]])
    # Ambiguous expressions such as 'twelve fifty' remain intact for reasoning.
    return match.group()


def normalise_text(text: str) -> str:
    """Create search text; leave ambiguous numbers and meaningful short replies intact."""
    text = re.sub(r"\b(?:[A-Za-z]_){2,}", lambda m: m.group().replace("_", ""), text)
    text = re.sub(r"\b(?:um+|uh+|hmm+|mm+)\b", " ", text, flags=re.IGNORECASE)
    # Currency context disambiguates 'twelve Euro fifty' from a list of numbers.
    currency = r"\b(" + "|".join(NUMBER_WORDS) + r")\s+euros?\s+(" + "|".join(NUMBER_WORDS) + r")\b"
    text = re.sub(currency, lambda m: f"{NUMBER_WORDS[m[1].lower()]}.{NUMBER_WORDS[m[2].lower()]:02d} EUR", text, flags=re.IGNORECASE)
    text = NUMBER_PATTERN.sub(_number, text)
    text = re.sub(r"\b(\d+) and a half\b", r"\1.5", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(\d+) point (\d)\b", r"\1.\2", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def build_ingestion(
    raw_dir: str | Path = RAW_DIR,
    parsed_dir: str | Path = PARSED_DIR,
    chunks_dir: str | Path = CHUNKS_DIR,
) -> dict[str, int]:
    """Parse and chunk the canonical input, preserving source files and raw text."""
    from app.ingestion.loader import TranscriptLoader
    from app.ingestion.parser import TranscriptParser
    from app.ingestion.chunker import TranscriptChunker

    raw_dir, parsed_dir, chunks_dir = map(project_path, (raw_dir, parsed_dir, chunks_dir))
    raw_dir, parsed_dir, chunks_dir = (path.resolve() for path in (raw_dir, parsed_dir, chunks_dir))
    for output_dir in (parsed_dir, chunks_dir):
        if output_dir.is_relative_to(raw_dir) or output_dir.is_relative_to(RAW_DIR.parent.resolve()):
            raise ValueError("Generated output must be outside the read-only raw data directory")
    if parsed_dir == chunks_dir:
        raise ValueError("Parsed meetings and chunks require separate output directories")
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Transcript directory not found: {raw_dir}")
    transcripts = TranscriptLoader(raw_dir).load_all()
    if not transcripts:
        raise ValueError(f"No readable transcripts found in {raw_dir}")
    expected = len(list(raw_dir.glob("*.txt")))
    if len(transcripts) != expected:
        raise ValueError(f"Loaded {len(transcripts)} of {expected} transcripts; check loader warnings")
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parser = TranscriptParser()
    chunker = TranscriptChunker()
    counts = {"meetings": 0, "utterances": 0, "chunks": 0}
    for transcript in transcripts:
        parsed = parser.parse(transcript)
        if not parsed["utterances"] or parsed["malformed_entries"] or parsed["unparsed_content"] or parsed["missing_metadata"] or parsed["warnings"]:
            raise ValueError(f"Transcript needs review: {parsed['meeting_id']}: {parsed['warnings']}")
        for turn in parsed["utterances"]:
            if parsed["original_transcript_text"][turn["start_char"]:turn["end_char"]] != turn["original_text"]:
                raise ValueError(f"Invalid source offsets: {parsed['meeting_id']}")
            turn["normalised_text"] = normalise_text(turn["original_text"])
        path = parsed_dir / f"{parsed['meeting_id']}.json"
        path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        chunk_file = chunker.chunk_and_save_meeting(parsed, chunks_dir)
        chunk_data = json.loads(chunk_file.read_text(encoding="utf-8"))
        counts["meetings"] += 1
        counts["utterances"] += len(parsed["utterances"])
        counts["chunks"] += chunk_data["total_chunks"]
    return counts
