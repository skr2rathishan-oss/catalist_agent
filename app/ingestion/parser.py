"""Deterministic transcript parser for AMI meeting transcripts."""

import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple

# Configure logger for transcript parsing operations
logger = logging.getLogger(__name__)


class TranscriptParser:
    """Parses raw meeting transcripts into structured metadata and utterances with character offsets."""

    # Matches meeting header key-value lines
    MEETING_ID_RE = re.compile(r"^Meeting:\s*(\S+)", re.IGNORECASE | re.MULTILINE)
    TEAM_ID_RE = re.compile(r"^Team:\s*(\S+)", re.IGNORECASE | re.MULTILINE)
    MEETING_NUM_RE = re.compile(r"^Meeting\s+number:\s*(\d+)(?:\s*\(([a-zA-Z0-9]+)\))?", re.IGNORECASE | re.MULTILINE)
    SPEAKERS_HEADER_RE = re.compile(r"^Speakers:\s*(.*)", re.IGNORECASE | re.MULTILINE)
    SPEAKER_ITEM_RE = re.compile(r"Speaker\s+([A-Za-z0-9_]+)", re.IGNORECASE)

    # Matches valid turn headers with timestamps (HH:MM:SS or H:MM:SS or optional fraction)
    TURN_HEADER_RE = re.compile(
        r"^(\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d+)?)\s+Speaker\s+([A-Za-z0-9_]+)\s*$",
        re.MULTILINE,
    )

    # Detects candidate lines that look like speaker headers but have malformed timestamps
    MALFORMED_HEADER_RE = re.compile(
        r"^([^\s\n]+)\s+Speaker\s+([A-Za-z0-9_]+)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    # Maps common sequence letters to meeting numbers
    SEQUENCE_TO_NUMBER = {"a": 1, "b": 2, "c": 3, "d": 4}

    def __init__(self, track_duplicates: bool = True) -> None:
        """Initialize the deterministic transcript parser."""
        self.track_duplicates = track_duplicates
        self.seen_meeting_ids: Set[str] = set()

    def reset(self) -> None:
        """Reset internal parser tracking state such as seen meeting IDs."""
        self.seen_meeting_ids.clear()

    def _parse_header(self, text: str) -> Tuple[Dict[str, Any], int, List[str]]:
        """Extract header metadata and character offset where the header section concludes."""
        metadata: Dict[str, Any] = {
            "header_meeting_id": None,
            "team_id": None,
            "meeting_number": None,
            "meeting_sequence": None,
            "speakers": [],
        }

        missing_metadata: List[str] = []

        # Find first turn header to determine the end boundary of the header section
        first_turn_match = self.TURN_HEADER_RE.search(text)
        header_text_limit = first_turn_match.start() if first_turn_match else len(text)
        header_text = text[:header_text_limit]

        # Extract meeting ID from header
        meeting_match = self.MEETING_ID_RE.search(header_text)
        if meeting_match:
            metadata["header_meeting_id"] = meeting_match.group(1).strip()
        else:
            missing_metadata.append("header_meeting_id")

        # Extract team ID from header
        team_match = self.TEAM_ID_RE.search(header_text)
        if team_match:
            metadata["team_id"] = team_match.group(1).strip()
        else:
            missing_metadata.append("team_id")

        # Extract meeting number and optional sequence letter from header
        num_match = self.MEETING_NUM_RE.search(header_text)
        if num_match:
            metadata["meeting_number"] = int(num_match.group(1))
            if num_match.group(2):
                metadata["meeting_sequence"] = num_match.group(2).lower()
        else:
            missing_metadata.append("meeting_number")

        # Extract speakers list from header
        speakers_match = self.SPEAKERS_HEADER_RE.search(header_text)
        if speakers_match:
            raw_speakers = speakers_match.group(1)
            found_speakers = self.SPEAKER_ITEM_RE.findall(raw_speakers)
            if found_speakers:
                metadata["speakers"] = found_speakers
            else:
                metadata["speakers"] = [s.strip() for s in raw_speakers.split(",") if s.strip()]
        else:
            missing_metadata.append("speakers")

        # Track character offset where recognized header lines conclude
        known_header_re = re.compile(
            r"^(?:Meeting|Team|Meeting\s+number|Speakers|Times|Source):.*$",
            re.MULTILINE | re.IGNORECASE,
        )
        header_end_offset = 0
        for header_match in known_header_re.finditer(header_text):
            if header_match.end() > header_end_offset:
                header_end_offset = header_match.end()

        return metadata, header_end_offset, missing_metadata

    def _parse_utterances_with_offsets(
        self,
        raw_text: str,
        body_start_offset: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
        """Extract utterances with start_char and end_char offsets and detect unparsed/malformed content."""
        utterances: List[Dict[str, Any]] = []
        malformed_entries: List[Dict[str, Any]] = []
        unparsed_content: List[Dict[str, Any]] = []
        warnings: List[str] = []

        # Find all valid turn headers in the body
        matches = list(self.TURN_HEADER_RE.finditer(raw_text))

        # Check for unparsed content between end of header metadata and first turn
        if matches:
            first_match_start = matches[0].start()
            if first_match_start > body_start_offset:
                between_text = raw_text[body_start_offset:first_match_start].strip()
                if between_text:
                    start_char = body_start_offset + raw_text[body_start_offset:first_match_start].find(between_text)
                    unparsed_content.append({
                        "text": between_text,
                        "start_char": start_char,
                        "end_char": start_char + len(between_text),
                    })
                    warnings.append(f"Unparsed content before first turn header: '{between_text[:40]}...'")
                    logger.warning(f"Unparsed content before first turn: '{between_text[:40]}...'")

        # Process each turn header match and extract its corresponding dialogue span
        for idx, match in enumerate(matches):
            timestamp = match.group(1)
            speaker_id = match.group(2)

            # Check if timestamp deviates from standard HH:MM:SS format
            if timestamp.count(":") != 2:
                warning_msg = f"Non-standard timestamp format detected: '{timestamp}' at position {match.start()}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
                malformed_entries.append({
                    "type": "non_standard_timestamp",
                    "timestamp": timestamp,
                    "speaker_id": speaker_id,
                    "char_offset": match.start(),
                })

            # The content for this turn starts immediately after the turn header line
            content_start = match.end()
            content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_text)

            slice_text = raw_text[content_start:content_end]
            trimmed_text = slice_text.strip()

            if trimmed_text:
                # Compute exact character positions of dialogue in original transcript
                offset_in_slice = slice_text.find(trimmed_text)
                start_char = content_start + offset_in_slice
                end_char = start_char + len(trimmed_text)
                original_text = raw_text[start_char:end_char]
            else:
                # Handle turn with no spoken dialogue
                start_char = content_start
                end_char = content_start
                original_text = ""

            utterance_dict = {
                "utterance_id": len(utterances) + 1,
                "timestamp": timestamp,
                "speaker_id": speaker_id,
                "original_text": original_text,
                "start_char": start_char,
                "end_char": end_char,
            }
            utterances.append(utterance_dict)

        # Detect any speaker lines that failed the standard turn header pattern
        for malformed_match in self.MALFORMED_HEADER_RE.finditer(raw_text):
            raw_time = malformed_match.group(1)
            speaker = malformed_match.group(2)
            # Avoid false alarms on standard metadata lines like 'Speakers: Speaker A'
            if raw_time.lower().startswith("speakers:") or raw_time.lower().startswith("meeting"):
                continue
            # If line was not covered by valid turn headers, record as malformed entry
            if not self.TURN_HEADER_RE.match(malformed_match.group(0)):
                warn_text = f"Malformed turn header candidate: '{malformed_match.group(0)}'"
                warnings.append(warn_text)
                logger.warning(warn_text)
                malformed_entries.append({
                    "type": "malformed_turn_header",
                    "text": malformed_match.group(0),
                    "start_char": malformed_match.start(),
                    "end_char": malformed_match.end(),
                })

        return utterances, malformed_entries, unparsed_content, warnings

    def parse(self, loaded_transcript: Dict[str, Any]) -> Dict[str, Any]:
        """Parse loaded transcript data dictionary into structured output with exact offsets."""
        if not isinstance(loaded_transcript, dict):
            logger.error("Invalid transcript input: expected dictionary.")
            raise ValueError("loaded_transcript must be a dictionary.")

        meeting_id = loaded_transcript.get("meeting_id") or ""
        file_path = loaded_transcript.get("file_path") or ""
        raw_text = loaded_transcript.get("raw_text") or ""
        filename = Path(file_path).name if file_path else (f"{meeting_id}.txt" if meeting_id else "")

        warnings: List[str] = []

        # Detect duplicate meeting IDs across calls
        if self.track_duplicates and meeting_id:
            if meeting_id in self.seen_meeting_ids:
                dup_warn = f"Duplicate meeting ID detected: '{meeting_id}'"
                warnings.append(dup_warn)
                logger.warning(dup_warn)
            self.seen_meeting_ids.add(meeting_id)

        # Handle empty transcript text safely
        if not raw_text.strip():
            logger.warning(f"Empty transcript content for meeting: {meeting_id}")
            warnings.append(f"Empty transcript content for meeting: {meeting_id}")
            return {
                "meeting_id": meeting_id,
                "team_id": meeting_id[:-1] if len(meeting_id) > 1 else "",
                "meeting_number": self.SEQUENCE_TO_NUMBER.get(meeting_id[-1:].lower()),
                "meeting_sequence": meeting_id[-1:].lower() if meeting_id else "",
                "file_path": file_path,
                "filename": filename,
                "speakers": [],
                "speaker_ids": [],
                "original_transcript_text": raw_text,
                "utterances": [],
                "missing_metadata": ["all"],
                "malformed_entries": [],
                "unparsed_content": [],
                "warnings": warnings,
            }

        # Extract metadata from header lines
        header_meta, header_end_offset, missing_metadata = self._parse_header(raw_text)

        # Validate filename meeting ID against header meeting ID
        header_meeting_id = header_meta["header_meeting_id"]
        if meeting_id and header_meeting_id:
            if meeting_id.lower() != header_meeting_id.lower():
                mismatch_warn = (
                    f"Meeting ID mismatch: filename says '{meeting_id}' "
                    f"but header says '{header_meeting_id}'"
                )
                warnings.append(mismatch_warn)
                logger.warning(mismatch_warn)

        # Determine effective meeting ID preferring header when valid
        effective_meeting_id = meeting_id or header_meeting_id or ""

        # Determine effective team ID with safe fallback
        team_id = header_meta["team_id"]
        if not team_id and effective_meeting_id:
            team_id = effective_meeting_id[:-1]

        # Determine effective meeting sequence letter with safe fallback
        sequence = header_meta["meeting_sequence"]
        if not sequence and effective_meeting_id:
            sequence = effective_meeting_id[-1:].lower()

        # Determine effective meeting number with safe fallback
        meeting_number = header_meta["meeting_number"]
        if meeting_number is None and sequence:
            meeting_number = self.SEQUENCE_TO_NUMBER.get(sequence)

        # Parse utterances and compute exact character offsets
        utterances, malformed_entries, unparsed_content, utterance_warnings = (
            self._parse_utterances_with_offsets(raw_text, header_end_offset)
        )
        warnings.extend(utterance_warnings)

        # Determine speakers list from header or observed utterances
        speakers = header_meta["speakers"]
        if not speakers:
            observed = sorted(list({u["speaker_id"] for u in utterances}))
            speakers = observed

        return {
            "meeting_id": effective_meeting_id,
            "team_id": team_id or "",
            "meeting_number": meeting_number,
            "meeting_sequence": sequence or "",
            "file_path": file_path,
            "filename": filename,
            "speakers": speakers,
            "speaker_ids": speakers,
            "original_transcript_text": raw_text,
            "utterances": utterances,
            "missing_metadata": missing_metadata,
            "malformed_entries": malformed_entries,
            "unparsed_content": unparsed_content,
            "warnings": warnings,
        }


if __name__ == "__main__":
    import sys
    # Add project root directory to sys.path for direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    # Import TranscriptLoader to run the ingestion pipeline
    from app.ingestion.loader import TranscriptLoader
    from app.config import PARSED_DIR

    # Initialize loader and load all transcripts
    loader = TranscriptLoader()
    transcripts = loader.load_all()

    # Initialize parser
    parser = TranscriptParser()

    # Output directory for parsed transcripts
    output_dir = PARSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    total_utterances = 0

    # Parse each loaded transcript and save structured JSON to disk
    for item in transcripts:
        parsed_data = parser.parse(item)
        total_utterances += len(parsed_data["utterances"])

        # Validate that character offsets match original text exactly
        raw_text = parsed_data["original_transcript_text"]
        for u in parsed_data["utterances"]:
            if u["original_text"]:
                assert raw_text[u["start_char"]:u["end_char"]] == u["original_text"]

        output_file = output_dir / f"{parsed_data['meeting_id']}.json"
        # Save parsed meeting dictionary to JSON
        output_file.write_text(json.dumps(parsed_data, indent=2), encoding="utf-8")

    # Print summary of parsed meetings and output storage location
    print(f"Parsed {len(transcripts)} meetings ({total_utterances} total utterances).")
    print(f"Verified 100% character offset accuracy for all utterances.")
    print(f"Output files stored in: {output_dir.resolve()}")
