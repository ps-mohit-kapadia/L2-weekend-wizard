from __future__ import annotations

"""Run representative smoke scenarios for the Claude SDK Weekend Wizard path."""

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from claude_sdk_agent.smoke import run_smoke_scenarios


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Claude SDK smoke harness."""
    parser = argparse.ArgumentParser(
        description="Run smoke scenarios for the Claude SDK Weekend Wizard path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print scenario and configuration information without making a live Claude call.",
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = parse_args()
    if not args.dry_run and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "Claude SDK smoke requires ANTHROPIC_API_KEY in live mode. "
            "Use --dry-run to verify scenario/config shape without calling Claude."
        )

    results = await run_smoke_scenarios(dry_run=args.dry_run)
    print(json.dumps({"mode": "dry-run" if args.dry_run else "live", "scenarios": results}, indent=2))


def main() -> None:
    """Entry point for the Claude SDK smoke harness."""
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
