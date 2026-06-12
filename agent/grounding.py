from __future__ import annotations

"""Grounding helpers for tool-result parsing and final answer composition."""

from typing import Any, Iterable, List

from schemas.tools import (
    BookResults,
    DogResult,
    GeoResult,
    JokeResult,
    ToolError,
    TriviaResult,
    WeatherResult,
)


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


def _empty_semantics() -> dict[str, str | None]:
    return {
        "planner_line": None,
        "detail_line": None,
        "fact_line": None,
    }


def _city_lookup_semantics(args: dict[str, Any], payload: Any) -> dict[str, str | None]:
    if isinstance(payload, ToolError):
        line = "- City Lookup: unavailable ({})".format(payload.details or payload.error)
        return {
            "planner_line": line,
            "detail_line": line,
            "fact_line": None,
        }
    if isinstance(payload, GeoResult):
        line = f"- City Lookup: {payload.city}: {payload.latitude}, {payload.longitude}"
        return {
            "planner_line": line,
            "detail_line": line,
            "fact_line": f"Resolved {payload.city} to {payload.latitude}, {payload.longitude}.",
        }
    return _empty_semantics()


def _weather_semantics(args: dict[str, Any], payload: Any) -> dict[str, str | None]:
    if isinstance(payload, ToolError):
        line = f"- Weather: {_coords_label(args)} unavailable ({payload.details or payload.error})"
        return {
            "planner_line": line,
            "detail_line": line,
            "fact_line": None,
        }
    if isinstance(payload, WeatherResult) and payload.temperature is not None:
        detail_line = (
            f"- Weather: {_coords_label(args)}: "
            f"{payload.temperature}{payload.temperature_unit or ''}, "
            f"{payload.weather_summary or 'current conditions'}"
        )
        return {
            "planner_line": detail_line,
            "detail_line": detail_line,
            "fact_line": (
                f"Weather for {_coords_label(args)}: "
                f"{payload.temperature}{payload.temperature_unit or ''}, "
                f"{payload.weather_summary or 'current conditions'}."
            ),
        }
    return _empty_semantics()


def _book_semantics(args: dict[str, Any], payload: Any) -> dict[str, str | None]:
    if isinstance(payload, ToolError):
        line = "- Books: unavailable ({})".format(payload.details or payload.error)
        return {
            "planner_line": line,
            "detail_line": line,
            "fact_line": None,
        }
    if isinstance(payload, BookResults) and payload.results:
        titles = [
            f"{book.title} by {book.author}"
            for book in payload.results[:2]
            if book.title
        ]
        if titles:
            rendered_titles = "; ".join(titles)
            line = f"- Books: {rendered_titles}"
            return {
                "planner_line": line,
                "detail_line": line,
                "fact_line": f"Book ideas for {payload.topic}: {rendered_titles}.",
            }
    return _empty_semantics()


def _joke_semantics(args: dict[str, Any], payload: Any) -> dict[str, str | None]:
    if isinstance(payload, ToolError):
        line = "- Joke: unavailable ({})".format(payload.details or payload.error)
        return {
            "planner_line": line,
            "detail_line": line,
            "fact_line": None,
        }
    if isinstance(payload, JokeResult):
        line = f"- Joke: {payload.joke}"
        return {
            "planner_line": line,
            "detail_line": line,
            "fact_line": f"Joke: {payload.joke}",
        }
    return _empty_semantics()


def _dog_semantics(args: dict[str, Any], payload: Any) -> dict[str, str | None]:
    if isinstance(payload, ToolError):
        line = "- Dog Pic: unavailable ({})".format(payload.details or payload.error)
        return {
            "planner_line": line,
            "detail_line": line,
            "fact_line": None,
        }
    if isinstance(payload, DogResult):
        return {
            "planner_line": "- Dog Pic: fetched one dog image",
            "detail_line": f"- Dog Pic: {payload.image_url}",
            "fact_line": f"Dog pic: {payload.image_url}",
        }
    return _empty_semantics()


def _trivia_semantics(args: dict[str, Any], payload: Any) -> dict[str, str | None]:
    if isinstance(payload, ToolError):
        line = "- Trivia: unavailable ({})".format(payload.details or payload.error)
        return {
            "planner_line": line,
            "detail_line": line,
            "fact_line": None,
        }
    if isinstance(payload, TriviaResult):
        choices = payload.incorrect_answers + [payload.correct_answer]
        line = f"- Trivia: {payload.question} Choices: {', '.join(choices)}"
        return {
            "planner_line": line,
            "detail_line": line,
            "fact_line": f"Trivia: {payload.question} Choices: {', '.join(choices)}.",
        }
    return _empty_semantics()


GROUNDING_RENDERERS = {
    "city_to_coords": _city_lookup_semantics,
    "get_weather": _weather_semantics,
    "book_recs": _book_semantics,
    "random_joke": _joke_semantics,
    "random_dog": _dog_semantics,
    "trivia": _trivia_semantics,
}


def _validate_grounding_renderers() -> None:
    for tool_name, renderer in GROUNDING_RENDERERS.items():
        if not tool_name:
            raise RuntimeError("Grounding renderer registry contains an empty tool name")
        if not callable(renderer):
            raise RuntimeError(f"Grounding renderer for {tool_name} must be callable")


_validate_grounding_renderers()


def extract_step_semantics(tool_name: str, args: dict[str, Any], payload: Any) -> dict[str, str | None]:
    """Extract shared semantic text fragments from one parsed execution step."""
    renderer = GROUNDING_RENDERERS.get(tool_name)
    if renderer is None:
        return _empty_semantics()
    return renderer(args, payload)


def build_grounded_items(user_prompt: str, steps: Iterable[Any]) -> List[Any]:
    """Return simple objects with title/detail from execution steps."""
    items: List[Any] = []
    lowered = user_prompt.lower()

    for step in steps:
        if getattr(step, "kind", "") not in {"tool_call", "tool_invalid"}:
            continue
        tool_name, args, payload = _step_fields(step)
        detail_line = extract_step_semantics(tool_name, args, payload)["detail_line"]
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
        line = extract_step_semantics(tool_name, args, payload)["planner_line"]
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
        semantics = extract_step_semantics(tool_name, args, payload)
        detail_line = semantics["detail_line"]
        if detail_line:
            detail_lines.append(detail_line)
        fact_line = semantics["fact_line"]
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
