from __future__ import annotations

"""Execution-time argument normalization helpers for Weekend Wizard."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from schemas.agent import ExecutionPlan


@dataclass
class ExecutionStateSnapshot:
    """Minimal execution state needed for tool-arg guardrails."""

    user_prompt: str
    plan: ExecutionPlan
    resolved_coords: Optional[Tuple[float, float]]


def normalize_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    state: ExecutionStateSnapshot,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Normalize and repair planned tool args before execution."""
    args = dict(args or {})

    if tool_name == "city_to_coords":
        city = args.get("city") or (state.plan.location.city if state.plan.location else None)
        if not city:
            return None, "city is required"
        return {"city": str(city)}, None

    if tool_name == "get_weather":
        latitude = args.get("latitude")
        longitude = args.get("longitude")
        if latitude is None or longitude is None:
            coords = state.resolved_coords
            if coords is not None:
                latitude, longitude = coords
            elif state.plan.location and state.plan.location.latitude is not None and state.plan.location.longitude is not None:
                latitude, longitude = state.plan.location.latitude, state.plan.location.longitude
        if latitude is None or longitude is None:
            return None, "latitude and longitude are required"
        try:
            return {"latitude": float(latitude), "longitude": float(longitude)}, None
        except (TypeError, ValueError):
            return None, "latitude and longitude must be numeric"

    if tool_name == "book_recs":
        topic = args.get("topic") or args.get("param") or state.plan.book_topic
        limit = args.get("limit") or 3
        if not topic:
            return None, "topic is required"
        try:
            safe_limit = max(1, min(int(limit), 10))
        except (TypeError, ValueError):
            safe_limit = 3
        return {"topic": str(topic), "limit": safe_limit}, None

    if tool_name in {"random_joke", "random_dog", "trivia"}:
        return {}, None

    return args, None
