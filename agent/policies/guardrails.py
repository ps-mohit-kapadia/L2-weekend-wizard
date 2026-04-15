from __future__ import annotations

"""Guardrails that prevent the live agent from finishing too early."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple


def parse_coords(text: str) -> Optional[Tuple[float, float]]:
    """Parse latitude and longitude coordinates from free-form text."""
    match = re.search(r"\((-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\)", text)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def infer_city(text: str) -> Optional[str]:
    """Infer a city phrase from a prompt when explicit coordinates are absent."""
    patterns = [
        r"\bin ([A-Z][a-zA-Z]+(?:[ \-][A-Z][a-zA-Z]+)*)",
        r"\bfor ([A-Z][a-zA-Z]+(?:[ \-][A-Z][a-zA-Z]+)*)",
        r"\bat ([A-Z][a-zA-Z]+(?:[ \-][A-Z][a-zA-Z]+)*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            if "(" not in candidate and ")" not in candidate:
                return candidate

    return None


def requested_tools(prompt: str) -> Set[str]:
    """Infer which tool categories the user is explicitly asking for."""
    lowered = prompt.lower()
    coords = parse_coords(prompt)
    requested: Set[str] = set()

    if coords is not None or infer_city(prompt) is not None or "weather" in lowered or "temperature" in lowered:
        requested.add("get_weather")
    if "book" in lowered or "read" in lowered or "theme" in lowered:
        requested.add("book_recs")
    if "joke" in lowered:
        requested.add("random_joke")
    if "dog" in lowered:
        requested.add("random_dog")
    if "trivia" in lowered:
        requested.add("trivia")

    return requested


def missing_requested_tools(prompt: str, payloads: Dict[str, Any]) -> List[str]:
    """Return requested tools that have not yet produced payloads."""
    return [
        tool_name
        for tool_name in requested_tools(prompt)
        if tool_name not in payloads
    ]
