from __future__ import annotations

"""Run the first Claude Agent SDK Weekend Wizard slice."""

import argparse
import asyncio
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from claude_sdk_agent.runner import run_claude_sdk_prompt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the SDK runner."""
    parser = argparse.ArgumentParser(
        description="Run the first Claude Agent SDK Weekend Wizard slice.",
    )
    parser.add_argument(
        "--prompt",
        default="Tell me a joke.",
        help="Prompt to send to the Claude Agent SDK runner.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the configured SDK path without making a live Claude call.",
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = parse_args()
    result = await run_claude_sdk_prompt(args.prompt, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


def main() -> None:
    """Entry point for the SDK runner script."""
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
