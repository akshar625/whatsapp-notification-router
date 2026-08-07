#!/usr/bin/env python3
"""Conventional entry point (AGENTS.md 6.4).

Runs the full pipeline over `dataset/messages.csv`, writes `output.csv`, then
validates the result and prints the distribution report.

    python code/main.py                # media enrichment (cached) + routing
    python code/main.py --skip-media   # routing only, if media is already cached

Both stages are idempotent: media artifacts are cached by SHA-256 of the file
bytes and decisions are cached per message_id, so a re-run costs nothing and
resumes after an interruption.

Requires GEMINI_API_KEY (vision) and ANTHROPIC_API_KEY (decisions) in the
environment or a .env file. See README.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser(description="Message Notification Router")
    ap.add_argument("--skip-media", action="store_true",
                    help="skip Stage 1 and use whatever media artifacts are cached")
    ap.add_argument("--force", action="store_true",
                    help="ignore caches and recompute everything")
    a = ap.parse_args()

    from config import CONFIG
    CONFIG.paths.ensure()

    if not a.skip_media:
        from media import enrich_all
        print("=== Stage 1: media enrichment (ASR + vision) ===")
        enrich_all(force=a.force)

    import predict
    print("\n=== Stages 0/2/3/4: context, retrieval, decision, constraints ===")
    rows = predict.run(force=a.force)
    ok = predict.validate(rows)
    predict.report(rows)
    predict.flags(rows)
    print(f"\noutput -> {CONFIG.paths.output_csv}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
