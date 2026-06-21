from __future__ import annotations

"""Boundary rendering for typed tool results."""

from typing import Any, List

from schemas.tools import (
    BookArgs,
    BookResults,
    DogResult,
    GeoResult,
    JokeResult,
    ToolError,
    ToolArgs,
    TriviaResult,
    WeatherArgs,
    WeatherResult,
    dump_tool_args,
)


def _step_fields(step: Any) -> tuple[str, ToolArgs | dict[str, Any], Any]:
    tool_name = getattr(step, "tool_name", "")
    args = getattr(step, "normalized_args", None) or getattr(step, "raw_args", None) or {}
    return tool_name, args, getattr(step, "parsed_payload", None)


def _coords_label(args: ToolArgs | dict[str, Any]) -> str:
    if isinstance(args, WeatherArgs):
        latitude = args.latitude
        longitude = args.longitude
    else:
        dumped = dump_tool_args(args)
        latitude = dumped.get("latitude")
        longitude = dumped.get("longitude")
    if latitude is None or longitude is None:
        return "requested location"
    return f"{latitude}, {longitude}"


def _tool_line(tool_name: str, args: ToolArgs | dict[str, Any], payload: Any) -> str | None:
    if isinstance(payload, ToolError):
        detail = payload.details or payload.error
        if tool_name == "city_to_coords":
            return f"- City Lookup: unavailable ({detail})"
        if tool_name == "get_weather":
            return f"- Weather: {_coords_label(args)} unavailable ({detail})"
        if tool_name == "book_recs":
            return f"- Books: unavailable ({detail})"
        if tool_name == "random_joke":
            return f"- Joke: unavailable ({detail})"
        if tool_name == "random_dog":
            return f"- Dog Pic: unavailable ({detail})"
        if tool_name == "trivia":
            return f"- Trivia: unavailable ({detail})"
        return f"- {tool_name}: unavailable ({detail})"

    if isinstance(payload, GeoResult):
        return f"- City Lookup: {payload.city}: {payload.latitude}, {payload.longitude}"
    if isinstance(payload, WeatherResult) and payload.temperature is not None:
        summary = payload.weather_summary or "current conditions"
        return f"- Weather: {_coords_label(args)}: {payload.temperature}{payload.temperature_unit or ''}, {summary}"
    if isinstance(payload, BookResults) and payload.results:
        requested_limit = args.limit if isinstance(args, BookArgs) else payload.count or len(payload.results)
        titles = [
            f"{book.title} by {book.author}"
            for book in payload.results[:requested_limit]
            if book.title
        ]
        if titles:
            return "- Books: " + "; ".join(titles)
    if isinstance(payload, JokeResult):
        return f"- Joke: {payload.joke}"
    if isinstance(payload, DogResult):
        return f"- Dog Pic: {payload.image_url}"
    if isinstance(payload, TriviaResult):
        choices = ", ".join(payload.incorrect_answers + [payload.correct_answer])
        return f"- Trivia: {payload.question} Choices: {choices}. Answer: {payload.correct_answer}"
    return None


def _tool_sentence(tool_name: str, args: ToolArgs | dict[str, Any], payload: Any) -> str | None:
    if isinstance(payload, GeoResult):
        return f"Resolved {payload.city} to {payload.latitude}, {payload.longitude}."
    if isinstance(payload, WeatherResult) and payload.temperature is not None:
        summary = payload.weather_summary or "current conditions"
        return f"Weather for {_coords_label(args)}: {payload.temperature}{payload.temperature_unit or ''}, {summary}."
    if isinstance(payload, BookResults) and payload.results:
        requested_limit = args.limit if isinstance(args, BookArgs) else payload.count or len(payload.results)
        titles = [
            f"{book.title} by {book.author}"
            for book in payload.results[:requested_limit]
            if book.title
        ]
        if titles:
            return f"Book ideas for {payload.topic}: {'; '.join(titles)}."
    if isinstance(payload, JokeResult):
        return f"Joke: {payload.joke}"
    if isinstance(payload, DogResult):
        return f"Dog pic: {payload.image_url}"
    if isinstance(payload, TriviaResult):
        choices = ", ".join(payload.incorrect_answers + [payload.correct_answer])
        return f"Trivia: {payload.question} Choices: {choices}. Answer: {payload.correct_answer}."
    return None


def _fact_prefix(tool_name: str) -> str:
    if tool_name == "city_to_coords":
        return "city_lookup"
    if tool_name == "get_weather":
        return "weather"
    if tool_name == "book_recs":
        return "books"
    if tool_name == "random_joke":
        return "joke"
    if tool_name == "random_dog":
        return "dog_pic"
    if tool_name == "trivia":
        return "trivia"
    return tool_name


def render_tool_feedback(tool_name: str, args: ToolArgs | dict[str, Any], payload: Any) -> str:
    """Render one typed tool result for the next ReAct planner step."""
    return _tool_line(tool_name, args, payload) or f"- {tool_name}: completed"


def render_reflection_observations(
    user_prompt: str,
    steps: List[Any],
) -> List[str]:
    """Render typed tool results for the reflection prompt."""
    rendered: List[str] = []
    counts: dict[str, int] = {}
    for step in steps:
        if getattr(step, "kind", "") not in {"tool_call", "tool_invalid"}:
            continue
        tool_name, args, payload = _step_fields(step)
        line = _tool_line(tool_name, args, payload)
        if not line:
            continue
        prefix = _fact_prefix(tool_name)
        counts[prefix] = counts.get(prefix, 0) + 1
        rendered.append(f"[{prefix}:{counts[prefix]}] {line}")
    if not rendered and "weekend" in user_prompt.lower():
        rendered.append("[detail:1] - Detail: Try a cozy cafe stop, a short walk, and a relaxing book session this weekend.")
    return rendered


def compose_grounded_answer_from_steps(
    user_prompt: str,
    answer: str,
    steps: List[Any],
) -> str:
    """Compose the grounded draft directly from typed execution-step payloads."""
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
        line = _tool_line(tool_name, args, payload)
        sentence = _tool_sentence(tool_name, args, payload)
        if line:
            detail_lines.append(line)
        if sentence:
            fact_lines.append(sentence)

    if not detail_lines and not fact_lines:
        return answer

    lowered = user_prompt.lower()
    is_plan_request = any(word in lowered for word in ("plan", "weekend", "saturday", "sunday"))

    if len(tool_steps) > 1 or is_plan_request:
        intro = "Weekend Wizard Plan" if is_plan_request else "Weekend Wizard Results"
        outro = ["Enjoy the vibe and follow the links if something catches your eye."] if is_plan_request else []
        return "\n".join([intro, *detail_lines, *outro])

    if len(fact_lines) == 1:
        return fact_lines[0]
    if fact_lines:
        return " ".join(fact_lines)
    return answer
