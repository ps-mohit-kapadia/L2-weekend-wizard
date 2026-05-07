from __future__ import annotations

"""Smoke scenarios and harness helpers for the Claude SDK Weekend Wizard path."""

from dataclasses import dataclass
from typing import Any

from claude_sdk_agent.runner import run_claude_sdk_prompt


@dataclass(frozen=True)
class SmokeScenario:
    """One representative Claude SDK smoke scenario."""

    label: str
    prompt: str


def get_smoke_scenarios() -> list[SmokeScenario]:
    """Return the representative Claude SDK smoke scenarios."""
    return [
        SmokeScenario(
            label="joke-only",
            prompt="Tell me a joke.",
        ),
        SmokeScenario(
            label="weekend-multi-tool",
            prompt="Plan a cozy Saturday in New York with today's weather, 3 mystery books, one joke, and a dog pic.",
        ),
    ]


async def run_smoke_scenarios(*, dry_run: bool) -> list[dict[str, Any]]:
    """Run the configured Claude SDK smoke scenarios."""
    results: list[dict[str, Any]] = []
    for scenario in get_smoke_scenarios():
        result = await run_claude_sdk_prompt(scenario.prompt, dry_run=dry_run)
        entry: dict[str, Any] = {
            "label": scenario.label,
            "prompt": scenario.prompt,
            "mode": result["mode"],
        }
        if dry_run:
            entry["config"] = result["config"]
        else:
            answer = result.get("answer", "")
            if not isinstance(answer, str) or not answer.strip():
                raise RuntimeError(f"Claude SDK smoke scenario '{scenario.label}' returned an empty answer.")
            entry["answer"] = answer
        results.append(entry)
    return results
