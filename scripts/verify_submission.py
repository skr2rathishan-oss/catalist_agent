"""Audit a saved local submission's IDs, fields, team scope, and exact citations."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import ANSWERS_FILE, PARSED_DIR
from app.submission import SUBMISSION_QUESTIONS_FILE, audit_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ANSWERS_FILE))
    parser.add_argument("--questions", default=str(SUBMISSION_QUESTIONS_FILE))
    parser.add_argument("--parsed-dir", default=str(PARSED_DIR))
    args = parser.parse_args()
    print(json.dumps(audit_submission(args.output, args.questions, args.parsed_dir), indent=2))


if __name__ == "__main__":
    main()
