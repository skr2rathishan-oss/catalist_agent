"""Transcript chunking module for grouping consecutive utterances into context windows."""

import json
import logging
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional

from app.config import CHUNK_TARGET, CHUNK_MAXIMUM, CHUNK_OVERLAP, PARSED_DIR, CHUNKS_DIR, project_path
from app.ingest import normalise_text

# Configure logger for chunking operations
logger = logging.getLogger(__name__)


class TranscriptChunker:
    """Groups consecutive meeting utterances into overlapping token-sized chunks."""

    def __init__(
        self,
        target_size: int = CHUNK_TARGET,
        maximum_size: int = CHUNK_MAXIMUM,
        overlap: int = CHUNK_OVERLAP,
        preserve_complete_utterances: bool = True,
        token_counter: Optional[Callable[[str], int]] = None,
    ) -> None:
        """Initialize chunker with token targets, overlap, and complete utterance policy."""
        if target_size <= 0 or maximum_size < target_size:
            raise ValueError("Require 0 < target_size <= maximum_size")
        if overlap < 0 or overlap >= target_size:
            raise ValueError("Require 0 <= overlap < target_size")
        self.target_size = target_size
        self.maximum_size = maximum_size
        self.overlap = overlap
        self.preserve_complete_utterances = preserve_complete_utterances
        self.token_counter = token_counter or self._default_token_counter

    def _split_long_turns(self, utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Split oversized turns into source spans only when explicitly requested."""
        if self.preserve_complete_utterances:
            return utterances
        result = []
        for turn in utterances:
            if self.token_counter(self._format_turn(turn)) <= self.maximum_size:
                result.append(turn)
                continue
            text = turn["original_text"]
            words = list(re.finditer(r"\S+", text))
            start = 0
            while start < len(words):
                end = start
                piece = dict(turn)
                while end < len(words):
                    piece["original_text"] = text[words[start].start():words[end].end()]
                    if self.token_counter(self._format_turn(piece)) > self.maximum_size:
                        break
                    end += 1
                if end == start:
                    raise ValueError("A word and its speaker header exceed maximum_size")
                left, right = words[start].start(), words[end - 1].end()
                piece["original_text"] = text[left:right]
                if "start_char" in turn:
                    piece["start_char"] = turn["start_char"] + left
                    piece["end_char"] = turn["start_char"] + right
                result.append(piece)
                start = end
        return result

    @staticmethod
    def _default_token_counter(text: str) -> int:
        """Estimate token count using word and punctuation matching."""
        if not text:
            return 0
        # Match words and punctuation marks similarly to BPE subword tokenizers
        return len(re.findall(r"\w+|[^\w\s]", text))

    def _format_turn(self, utterance: Dict[str, Any]) -> str:
        """Format a single utterance with timestamp and speaker prefix."""
        timestamp = utterance.get("timestamp")
        speaker_id = utterance.get("speaker_id")
        original_text = utterance.get("original_text", "")

        # Include header prefix if both timestamp and speaker are available
        if timestamp and speaker_id:
            return f"{timestamp} Speaker {speaker_id}\n{original_text}"
        if speaker_id:
            return f"Speaker {speaker_id}\n{original_text}"
        return original_text

    def _build_chunk(
        self,
        meeting_id: str,
        team_id: str,
        chunk_index: int,
        chunk_utterances: List[Dict[str, Any]],
        parsed_meeting: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Construct structured chunk dictionary with all required metadata."""
        chunk_id = f"{meeting_id}_c{chunk_index:03d}"

        # Collect turn IDs and unique speaker IDs in chronological order
        turn_ids = list(dict.fromkeys(u.get("utterance_id") for u in chunk_utterances if u.get("utterance_id") is not None))
        unique_speakers = sorted(list({u.get("speaker_id") for u in chunk_utterances if u.get("speaker_id")}))

        # Extract boundary timestamps
        start_timestamp = chunk_utterances[0].get("timestamp", "")
        end_timestamp = chunk_utterances[-1].get("timestamp", "")

        # Format complete dialogue text combining all utterances in chunk
        formatted_turns = [self._format_turn(u) for u in chunk_utterances]
        original_text = "\n\n".join(formatted_turns)

        # Compute exact token count for the chunk
        token_count = self.token_counter(original_text)

        # Build chunk dictionary conforming to required fields
        chunk_record: Dict[str, Any] = {
            "chunk_id": chunk_id,
            "meeting_id": meeting_id,
            "team_id": team_id,
            "turn_ids": turn_ids,
            "speaker_ids": unique_speakers,
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "original_text": original_text,
            "normalised_text": normalise_text(original_text),
            "token_count": token_count,
        }

        # Include character offsets if available in source utterances
        if "start_char" in chunk_utterances[0]:
            chunk_record["start_char"] = chunk_utterances[0]["start_char"]
        if "end_char" in chunk_utterances[-1]:
            chunk_record["end_char"] = chunk_utterances[-1]["end_char"]

        # Preserve optional meeting metadata without inventing values
        if "meeting_sequence" in parsed_meeting and parsed_meeting["meeting_sequence"]:
            chunk_record["meeting_sequence"] = parsed_meeting["meeting_sequence"]
        if "meeting_number" in parsed_meeting and parsed_meeting["meeting_number"] is not None:
            chunk_record["meeting_number"] = parsed_meeting["meeting_number"]

        return chunk_record

    def chunk_meeting(self, parsed_meeting: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Group consecutive meeting utterances into overlapping chunks."""
        if not isinstance(parsed_meeting, dict):
            logger.error("Expected parsed_meeting to be a dictionary.")
            raise ValueError("parsed_meeting must be a dictionary.")

        meeting_id = parsed_meeting.get("meeting_id", "")
        team_id = parsed_meeting.get("team_id", "")
        utterances = self._split_long_turns(parsed_meeting.get("utterances", []))

        # Return empty list safely if meeting has no utterances
        if not utterances:
            logger.info(f"Meeting {meeting_id} has no utterances to chunk.")
            return []

        chunks: List[Dict[str, Any]] = []
        n = len(utterances)
        start_idx = 0
        chunk_index = 1

        while start_idx < n:
            chunk_utterances: List[Dict[str, Any]] = []
            chunk_tokens = 0
            curr_idx = start_idx

            # Accumulate consecutive utterances until target size is reached
            while curr_idx < n:
                u = utterances[curr_idx]
                formatted_u = self._format_turn(u)
                u_tokens = self.token_counter(formatted_u)

                # Always add first utterance even if it exceeds target size alone
                if not chunk_utterances:
                    chunk_utterances.append(u)
                    chunk_tokens += u_tokens
                    curr_idx += 1
                    # Stop if single utterance already reaches or exceeds target size
                    if chunk_tokens >= self.target_size:
                        break
                else:
                    # Append utterance if within target size
                    if chunk_tokens + u_tokens <= self.target_size:
                        chunk_utterances.append(u)
                        chunk_tokens += u_tokens
                        curr_idx += 1
                    # Allow completing utterance within maximum size buffer
                    elif chunk_tokens + u_tokens <= self.maximum_size:
                        chunk_utterances.append(u)
                        chunk_tokens += u_tokens
                        curr_idx += 1
                        break
                    # Otherwise close chunk to prevent exceeding maximum size
                    else:
                        break

            # Build and append current chunk record
            chunk_record = self._build_chunk(
                meeting_id=meeting_id,
                team_id=team_id,
                chunk_index=chunk_index,
                chunk_utterances=chunk_utterances,
                parsed_meeting=parsed_meeting,
            )
            chunks.append(chunk_record)
            chunk_index += 1

            # If all utterances have been processed, terminate chunking
            if curr_idx >= n:
                break

            if self.overlap == 0:
                start_idx = curr_idx
                continue

            # Calculate overlap starting point moving backward from last utterance
            overlap_tokens = 0
            overlap_idx = curr_idx - 1

            while overlap_idx > start_idx:
                prev_u = utterances[overlap_idx]
                prev_tokens = self.token_counter(self._format_turn(prev_u))
                if overlap_tokens + prev_tokens > self.overlap:
                    # Ensure at least one utterance overlaps if none collected yet
                    if overlap_tokens == 0 and overlap_idx > start_idx:
                        overlap_tokens += prev_tokens
                        overlap_idx -= 1
                    break
                overlap_tokens += prev_tokens
                overlap_idx -= 1

            # Advance start index ensuring strict forward progress
            next_start = max(overlap_idx + 1, start_idx + 1)
            start_idx = next_start

        logger.info(f"Generated {len(chunks)} chunks for meeting {meeting_id}")
        return chunks

    def chunk_and_save_meeting(self, parsed_meeting: Dict[str, Any], output_dir: Path) -> Path:
        """Chunk meeting and save structured JSON output file."""
        meeting_id = parsed_meeting.get("meeting_id", "unknown")
        chunks = self.chunk_meeting(parsed_meeting)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{meeting_id}.json"

        # Organize meeting chunk output conforming to specification
        meeting_output = {
            "meeting_id": meeting_id,
            "team_id": parsed_meeting.get("team_id", ""),
            "meeting_number": parsed_meeting.get("meeting_number"),
            "meeting_sequence": parsed_meeting.get("meeting_sequence", ""),
            "total_chunks": len(chunks),
            "chunks": chunks,
        }

        # Save formatted JSON file to output directory
        output_file.write_text(json.dumps(meeting_output, indent=2), encoding="utf-8")
        return output_file

    def process_all(
        self,
        input_dir: Path | str = PARSED_DIR,
        output_dir: Path | str = CHUNKS_DIR,
    ) -> int:
        """Process and chunk all parsed meeting files in input directory."""
        in_path = project_path(input_dir)
        out_path = project_path(output_dir)

        json_files = sorted(in_path.glob("*.json"))
        if not json_files:
            logger.warning(f"No parsed JSON files found in {in_path}")
            return 0

        successful = 0
        for file_path in json_files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                self.chunk_and_save_meeting(data, out_path)
                successful += 1
            except Exception as error:
                logger.error(f"Error processing {file_path.name}: {error}")

        logger.info(f"Successfully processed {successful} of {len(json_files)} meetings into {out_path}")
        return successful


if __name__ == "__main__":
    import sys
    # Add project root directory to sys.path for direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    # Initialize chunker with default configuration
    chunker = TranscriptChunker(
        target_size=400,
        maximum_size=500,
        overlap=50,
        preserve_complete_utterances=True,
    )

    # Process all parsed transcripts into processed chunks directory
    total_processed = chunker.process_all(
        input_dir=PARSED_DIR,
        output_dir=CHUNKS_DIR,
    )

    print(f"Total meetings chunked: {total_processed}")
    output_sample = CHUNKS_DIR / "ES2002a.json"
    if output_sample.exists():
        sample_data = json.loads(output_sample.read_text(encoding="utf-8"))
        first_chunk = sample_data["chunks"][0]
        print(f"Sample Meeting ID: {sample_data['meeting_id']}")
        print(f"Total Chunks: {sample_data['total_chunks']}")
        print(f"Chunk ID: {first_chunk['chunk_id']}")
        print(f"Turns: {first_chunk['turn_ids']}")
        print(f"Speakers: {first_chunk['speaker_ids']}")
        print(f"Tokens: {first_chunk['token_count']}")
        print(f"Preview (first 120 chars):\n{first_chunk['original_text'][:120]}...")

