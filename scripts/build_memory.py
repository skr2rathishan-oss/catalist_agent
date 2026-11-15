"""Build local, cited event memory from parsed meetings; API use needs --run."""

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MEMORY_CONTEXT_MAX_WORDS, MEMORY_DIR, PARSED_DIR, load_generation_settings
from app.generation import NvidiaGenerationClient
from app.memory_pipeline import extract_meeting, reconcile_team, turn_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", help="One team ID, for example TS3012")
    parser.add_argument("--meeting", help="One meeting ID, for example TS3012b")
    parser.add_argument("--max-meetings", type=int, help="Process only the first N meetings")
    parser.add_argument("--max-words", type=int, default=MEMORY_CONTEXT_MAX_WORDS)
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Generation output limit for structured extraction")
    parser.add_argument("--model", help="Generation model override")
    parser.add_argument("--fallback-model", help="Use this model only if a window has no valid records")
    parser.add_argument("--run", action="store_true", help="Make cached NVIDIA calls and write memory")
    parser.add_argument("--no-reconcile", action="store_true", help="Extract without linking replacements")
    parser.add_argument("--reconcile-only", action="store_true",
                        help="Link complete saved team memories without rerunning extraction")
    args = parser.parse_args()
    if args.team and not re.fullmatch(r"[A-Z]{2}\d{4}", args.team):
        parser.error("--team must be a corpus team ID")
    if args.meeting and not re.fullmatch(r"[A-Z]{2}\d{4}[a-d]", args.meeting):
        parser.error("--meeting must be a corpus meeting ID")
    if args.team and args.meeting and not args.meeting.startswith(args.team):
        parser.error("--team and --meeting disagree")
    if args.max_meetings is not None and args.max_meetings < 1:
        parser.error("--max-meetings must be positive")
    if args.reconcile_only and (args.no_reconcile or args.meeting or args.max_meetings):
        parser.error("--reconcile-only needs complete teams and cannot use --no-reconcile, --meeting, or --max-meetings")
    files = ([PARSED_DIR / f"{args.meeting}.json"] if args.meeting else
             sorted(PARSED_DIR.glob(f"{args.team or '*'}[a-d].json")))
    files = [file for file in files if file.is_file()]
    if args.max_meetings:
        files = files[:args.max_meetings]
    if not files:
        parser.error("No parsed meetings found; run scripts/build.py first")
    meetings = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    windows = sum(len(turn_windows(meeting, args.max_words)) for meeting in meetings)
    if not args.run:
        print(json.dumps({"dry_run": True, "meetings": len(meetings),
                          "extraction_windows": 0 if args.reconcile_only else windows, "api_calls": 0,
                          "note": "Reconciliation calls depend on extracted same-topic events. Use --run to proceed."},
                         indent=2))
        return
    client = NvidiaGenerationClient(replace(load_generation_settings(model=args.model),
                                            max_tokens=args.max_tokens))
    fallback = (NvidiaGenerationClient(replace(load_generation_settings(model=args.fallback_model),
                                               max_tokens=args.max_tokens))
                if args.fallback_model else None)
    results, failures = [], []
    for meeting in ([] if args.reconcile_only else meetings):
        try:
            result = extract_meeting(meeting, client, MEMORY_DIR, args.max_words, fallback)
            results.append(result)
            print(json.dumps(result), flush=True)
        except (OSError, ValueError, RuntimeError) as error:
            failure = {"meeting_id": meeting["meeting_id"],
                       "reason": f"{type(error).__name__}: {error}"}
            failures.append(failure)
            print(json.dumps(failure), flush=True)
    teams = sorted({meeting["team_id"] for meeting in meetings})
    links = []
    if not args.no_reconcile:
        for team in teams:
            expected = {meeting["meeting_id"] for meeting in meetings if meeting["team_id"] == team}
            missing = sorted(meeting_id for meeting_id in expected
                             if not (MEMORY_DIR / f"{meeting_id}.json").is_file())
            if missing:
                failure = {"team_id": team, "reason": "Missing saved memory: " + ", ".join(missing)}
                failures.append(failure)
                print(json.dumps(failure), flush=True)
                continue
            if any(item["meeting_id"].startswith(team) for item in failures):
                continue
            try:
                result = reconcile_team(team, client, PARSED_DIR, MEMORY_DIR)
                links.append(result)
                print(json.dumps(result), flush=True)
            except (OSError, ValueError, RuntimeError) as error:
                failure = {"team_id": team, "reason": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                print(json.dumps(failure), flush=True)
    summary = {"meetings": len(meetings) if args.reconcile_only else len(results),
               "records": (sum(row["records"] for row in links) if args.reconcile_only
                           else sum(row["records"] for row in results)),
               "rejected_records": sum(row["rejected_records"] for row in results),
               "links": sum(row["links"] for row in links), "failures": failures,
               "fallback_windows": sum(row["fallback_windows"] for row in results),
               "api_requests": client.api_requests + (fallback.api_requests if fallback else 0),
               "cache_hits": client.cache_hits + (fallback.cache_hits if fallback else 0)}
    MEMORY_DIR.joinpath("build_runs").mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = MEMORY_DIR / "build_runs" / f"memory_{stamp}.json"
    report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(summary, report=str(report)), indent=2))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
