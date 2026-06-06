from __future__ import annotations

"""Grounding helpers for tool-result parsing and final answer composition."""

import json
from typing import Any, Iterable, List

from schemas.tools import (
    BookResults,
    DogResult,
    GeoResult,
    JokeResult,
    ToolError,
    TriviaResult,
    WeatherResult,
    parse_tool_payload,
)


def parse_tool_payload_text(tool_name: str, payload_text: str) -> Any:
    """Parse one serialized tool payload into a typed payload when possible."""
    try:
        return parse_tool_payload(tool_name, json.loads(payload_text))
    except json.JSONDecodeError:
        return payload_text


def _step_fields(step: Any) -> tuple[str, dict[str, Any], Any]:
    tool_name = getattr(step, "tool_name", "")
    args = dict(
        getattr(step, "normalized_args", None)
        or getattr(step, "raw_args", None)
        or {}
    )
    return tool_name, args, getattr(step, "parsed_payload", None)


def _coords_label(args: dict[str, Any]) -> str:
    latitude = args.get("latitude")
    longitude = args.get("longitude")
    if latitude is None or longitude is None:
        return "requested location"
    return f"{latitude}, {longitude}"


def _detail_line(tool_name: str, args: dict[str, Any], payload: Any, *, compact: bool = False) -> str | None:
    if tool_name == "city_to_coords":
        if isinstance(payload, ToolError):
            return "- City Lookup: unavailable ({})".format(payload.details or payload.error)
        if isinstance(payload, GeoResult):
            return f"- City Lookup: {payload.city}: {payload.latitude}, {payload.longitude}"
        return None

    if tool_name == "get_weather":
        if isinstance(payload, ToolError):
            return f"- Weather: {_coords_label(args)} unavailable ({payload.details or payload.error})"
        if isinstance(payload, WeatherResult) and payload.temperature is not None:
            return (
                f"- Weather: {_coords_label(args)}: "
                f"{payload.temperature}{payload.temperature_unit or ''}, "
                f"{payload.weather_summary or 'current conditions'}"
            )
        return None

    if tool_name == "book_recs":
        if isinstance(payload, ToolError):
            return "- Books: unavailable ({})".format(payload.details or payload.error)
        if isinstance(payload, BookResults) and payload.results:
            titles = [
                f"{book.title} by {book.author}"
                for book in payload.results[:2]
                if book.title
            ]
            if titles:
                return f"- Books: {'; '.join(titles)}"
        return None

    if tool_name == "random_joke":
        if isinstance(payload, ToolError):
            return "- Joke: unavailable ({})".format(payload.details or payload.error)
        if isinstance(payload, JokeResult):
            return f"- Joke: {payload.joke}"
        return None

    if tool_name == "random_dog":
        if isinstance(payload, ToolError):
            return "- Dog Pic: unavailable ({})".format(payload.details or payload.error)
        if isinstance(payload, DogResult):
            return "- Dog Pic: fetched one dog image" if compact else f"- Dog Pic: {payload.image_url}"
        return None

    if tool_name == "trivia":
        if isinstance(payload, ToolError):
            return "- Trivia: unavailable ({})".format(payload.details or payload.error)
        if isinstance(payload, TriviaResult):
            choices = payload.incorrect_answers + [payload.correct_answer]
            return f"- Trivia: {payload.question} Choices: {', '.join(choices)}"
        return None

    return None


def _fact_line(tool_name: str, args: dict[str, Any], payload: Any) -> str | None:
    if tool_name == "city_to_coords" and isinstance(payload, GeoResult):
        return f"Resolved {payload.city} to {payload.latitude}, {payload.longitude}."
    if tool_name == "get_weather" and isinstance(payload, WeatherResult) and payload.temperature is not None:
        return (
            f"Weather for {_coords_label(args)}: "
            f"{payload.temperature}{payload.temperature_unit or ''}, "
            f"{payload.weather_summary or 'current conditions'}."
        )
    if tool_name == "book_recs" and isinstance(payload, BookResults) and payload.results:
        titles = [
            f"{book.title} by {book.author}"
            for book in payload.results[:2]
            if book.title
        ]
        if titles:
            return f"Book ideas for {payload.topic}: {'; '.join(titles)}."
    if tool_name == "random_joke" and isinstance(payload, JokeResult):
        return f"Joke: {payload.joke}"
    if tool_name == "random_dog" and isinstance(payload, DogResult):
        return f"Dog pic: {payload.image_url}"
    if tool_name == "trivia" and isinstance(payload, TriviaResult):
        choices = payload.incorrect_answers + [payload.correct_answer]
        return f"Trivia: {payload.question} Choices: {', '.join(choices)}."
    return None


def build_grounded_items(user_prompt: str, steps: Iterable[Any]) -> List[Any]:
    """Return simple objects with title/detail from execution steps."""
    items: List[Any] = []
    lowered = user_prompt.lower()

    for step in steps:
        if getattr(step, "kind", "") not in {"tool_call", "tool_invalid"}:
            continue
        tool_name, args, payload = _step_fields(step)
        detail_line = _detail_line(tool_name, args, payload, compact=False)
        if not detail_line:
            continue
        title, detail = detail_line[2:].split(": ", 1)
        items.append(type("GroundedCompat", (), {"title": title, "detail": detail})())

    if not items and "weekend" in lowered:
        items.append(type("GroundedCompat", (), {"title": "Detail", "detail": "Try a cozy cafe stop, a short walk, and a relaxing book session this weekend."})())

    return items


def render_compact_step_summaries(
    user_prompt: str,
    steps: List[Any],
) -> List[str]:
    """Render compact grounded step summaries from execution-step state."""
    rendered: List[str] = []
    for step in steps:
        if getattr(step, "kind", "") not in {"tool_call", "tool_invalid"}:
            continue
        tool_name, args, payload = _step_fields(step)
        line = _detail_line(tool_name, args, payload, compact=True)
        if line:
            rendered.append(line)
    if not rendered and "weekend" in user_prompt.lower():
        rendered.append("- Detail: Try a cozy cafe stop, a short walk, and a relaxing book session this weekend.")
    return rendered


def compose_grounded_answer_from_steps(
    user_prompt: str,
    answer: str,
    steps: List[Any],
) -> str:
    """Compose the final grounded answer directly from execution steps."""
    tool_steps = [
        step
        for step in steps
        if getattr(step, "kind", "") in {"tool_call", "tool_invalid"}
        and getattr(step, "tool_name", "")
    ]
    if not tool_steps:
        return answer

    detail_lines: List[str] = []
    fact_lines: List[str] = []
    for step in tool_steps:
        tool_name, args, payload = _step_fields(step)
        detail_line = _detail_line(tool_name, args, payload, compact=False)
        if detail_line:
            detail_lines.append(detail_line)
        fact_line = _fact_line(tool_name, args, payload)
        if fact_line:
            fact_lines.append(fact_line)

    if not detail_lines and not fact_lines:
        return answer

    lowered = user_prompt.lower()
    is_plan_request = any(word in lowered for word in ("plan", "weekend", "saturday", "sunday"))

    if len(tool_steps) > 1 or is_plan_request:
        intro = "Weekend Wizard Plan" if is_plan_request else "Weekend Wizard Results"
        body = detail_lines
        outro = ["Enjoy the vibe and follow the links if something catches your eye."] if is_plan_request else []
        return "\n".join([intro, *body, *outro])

    if len(fact_lines) == 1:
        return fact_lines[0]
    if fact_lines:
        return " ".join(fact_lines)
    return answer
