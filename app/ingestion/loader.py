"""Transcript loader module for meeting transcripts."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

from app.config import RAW_DIR, project_path

# Configure basic logging for visibility
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TranscriptLoader:
    """Loads raw meeting transcript files from a specified folder."""

    def __init__(self, data_dir: Union[str, Path] = RAW_DIR) -> None:
        """Initialize the loader with the transcript directory path."""
        self.data_dir = project_path(data_dir)

    def load_file(self, file_path: Path) -> Optional[Dict[str, str]]:
        """Load a single transcript file and extract its metadata and content."""
        file_path = project_path(file_path)
        # Ensure the file exists and is a file
        if not file_path.is_file():
            logger.warning(f"File not found: {file_path}")
            return None

        # Extract meeting ID from file stem (e.g., 'ES2002a')
        meeting_id = file_path.stem

        try:
            # Read file content with UTF-8 preserving raw text exactly
            raw_text = file_path.read_text(encoding="utf-8")
        except Exception as error:
            logger.error(f"Error reading {file_path.name}: {error}")
            return None

        # Check for empty file content safely
        if not raw_text.strip():
            logger.warning(f"Skipping empty file: {file_path.name}")
            return None

        return {
            "meeting_id": meeting_id,
            "file_path": str(file_path),
            "raw_text": raw_text,
        }

    def load_all(self) -> List[Dict[str, str]]:
        """Load all .txt transcript files found in the data directory."""
        # Handle missing folder safely
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return []

        if not self.data_dir.is_dir():
            logger.warning(f"Specified path is not a directory: {self.data_dir}")
            return []

        # Find all .txt files in the directory
        txt_files = sorted(self.data_dir.glob("*.txt"))

        if not txt_files:
            logger.warning(f"No .txt transcript files found in: {self.data_dir}")
            return []

        transcripts: List[Dict[str, str]] = []

        for file_path in txt_files:
            meeting_data = self.load_file(file_path)
            if meeting_data is not None:
                transcripts.append(meeting_data)

        logger.info(f"Successfully loaded {len(transcripts)} meeting transcripts from {self.data_dir}")
        return transcripts


if __name__ == "__main__":
    # Initialize the loader with the transcript directory
    loader = TranscriptLoader()

    # Load all transcripts
    transcripts = loader.load_all()

    # Inspect the loaded data
    print(f"Total meetings loaded: {len(transcripts)}")

    if transcripts:
        first_meeting = transcripts[0]
        print(f"Meeting ID: {first_meeting['meeting_id']}")
        print(f"File Path: {first_meeting['file_path']}")
        print(f"Text Preview:\n{first_meeting['raw_text'][:150]}...")
