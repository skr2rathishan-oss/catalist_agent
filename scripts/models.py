"""List provider text-model candidates, or probe one selected model's JSON output."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import GENERATION_PROVIDERS, MODEL_CHECKS_DIR
from app.generation import GenerationError
from app.providers import discover_models, key_status, probe_model


def main():
    """No bulk inference: --probe makes just one selected model check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=[*GENERATION_PROVIDERS, "all"], default="all")
    parser.add_argument("--probe", metavar="MODEL", help="Make a fresh small JSON request; requires one provider")
    args = parser.parse_args()
    if args.probe and args.provider == "all":
        parser.error("Choose one --provider with --probe")
    providers = list(GENERATION_PROVIDERS) if args.provider == "all" else [args.provider]
    report = {"time": datetime.now(timezone.utc).isoformat(), "providers": {}}
    configured = key_status()
    failed = False
    for provider in providers:
        if not configured[provider]:
            report["providers"][provider] = {"status": "key not configured"}
            continue
        try:
            rows = discover_models(provider)
            item = {"status": "catalog fetched; inference not yet verified", "models": rows}
            if args.probe:
                if args.probe not in [row["model"] for row in rows]:
                    raise ValueError("Selected model is absent from the filtered catalog; inspect IDs and pricing first")
                item["check"] = probe_model(provider, args.probe)
                failed = item["check"]["status"] != "JSON check passed"
            report["providers"][provider] = item
        except (ValueError, GenerationError, OSError) as error:
            report["providers"][provider] = {"status": "check failed", "reason": str(error)}
            failed = True
    MODEL_CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = MODEL_CHECKS_DIR / f"models_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(report, report_file=str(path)), ensure_ascii=False, indent=2))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
