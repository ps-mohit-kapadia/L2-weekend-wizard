from __future__ import annotations

"""Grounding helpers for tool-result parsing and final answer composition."""

from dataclasses import dataclass
from typing import Any, Iterable, List

from schemas.tools import (
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


@dataclass(frozen=True)
class GroundedFact:
    """One grounded fact projected from execution truth for answer/reflection use."""

    label: str
    display_text: str
    sentence: str | None = None
    required_evidence: tuple[str, ...] = ()
    status: str = "success"


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


def _empty_facts() -> list[GroundedFact]:
    return []


def _city_lookup_facts(args: ToolArgs | dict[str, Any], payload: Any) -> list[GroundedFact]:
    if isinstance(payload, ToolError):
        detail = payload.details or payload.error
        return [
            GroundedFact(
                label="City Lookup",
                display_text=f"- City Lookup: unavailable ({detail})",
                required_evidence=("unavailable", detail),
                status="failed",
            )
        ]
    if isinstance(payload, GeoResult):
        display = f"- City Lookup: {payload.city}: {payload.latitude}, {payload.longitude}"
        return [
            GroundedFact(
                label="City Lookup",
                display_text=display,
                sentence=f"Resolved {payload.city} to {payload.latitude}, {payload.longitude}.",
            )
        ]
    return _empty_facts()


def _weather_facts(args: ToolArgs | dict[str, Any], payload: Any) -> list[GroundedFact]:
    if isinstance(payload, ToolError):
        location = _coords_label(args)
        detail = payload.details or payload.error
        return [
            GroundedFact(
                label="Weather",
                display_text=f"- Weather: {location} unavailable ({detail})",
                required_evidence=("unavailable", detail),
                status="failed",
            )
        ]
    if isinstance(payload, WeatherResult) and payload.temperature is not None:
        location = _coords_label(args)
        display = (
            f"- Weather: {location}: "
            f"{payload.temperature}{payload.temperature_unit or ''}, "
            f"{payload.weather_summary or 'current conditions'}"
        )
        summary = payload.weather_summary or "current conditions"
        return [
            GroundedFact(
                label="Weather",
                display_text=display,
                sentence=(
                    f"Weather for {location}: "
                    f"{payload.temperature}{payload.temperature_unit or ''}, "
                    f"{summary}."
                ),
                required_evidence=(
                    f"{payload.temperature}{payload.temperature_unit or ''}",
                    summary,
                ),
            )
        ]
    return _empty_facts()


def _book_facts(args: ToolArgs | dict[str, Any], payload: Any) -> list[GroundedFact]:
    if isinstance(payload, ToolError):
        detail = payload.details or payload.error
        return [
            GroundedFact(
                label="Books",
                display_text=f"- Books: unavailable ({detail})",
                required_evidence=("unavailable", detail),
                status="failed",
            )
        ]
    if isinstance(payload, BookResults) and payload.results:
        titles = [
            f"{book.title} by {book.author}"
            for book in payload.results[:2]
            if book.title
        ]
        if titles:
            rendered_titles = "; ".join(titles)
            return [
                GroundedFact(
                    label="Books",
                    display_text=f"- Books: {rendered_titles}",
                    sentence=f"Book ideas for {payload.topic}: {rendered_titles}.",
                    required_evidence=tuple(titles),
                )
            ]
    return _empty_facts()


def _joke_facts(args: ToolArgs | dict[str, Any], payload: Any) -> list[GroundedFact]:
    if isinstance(payload, ToolError):
        detail = payload.details or payload.error
        return [
            GroundedFact(
                label="Joke",
                display_text=f"- Joke: unavailable ({detail})",
                required_evidence=("unavailable", detail),
                status="failed",
            )
        ]
    if isinstance(payload, JokeResult):
        return [
            GroundedFact(
                label="Joke",
                display_text=f"- Joke: {payload.joke}",
                sentence=f"Joke: {payload.joke}",
                required_evidence=(payload.joke,),
            )
        ]
    return _empty_facts()


def _dog_facts(args: ToolArgs | dict[str, Any], payload: Any) -> list[GroundedFact]:
    if isinstance(payload, ToolError):
        detail = payload.details or payload.error
        return [
            GroundedFact(
                label="Dog Pic",
                display_text=f"- Dog Pic: unavailable ({detail})",
                required_evidence=("unavailable", detail),
                status="failed",
            )
        ]
    if isinstance(payload, DogResult):
        return [
            GroundedFact(
                label="Dog Pic",
                display_text=f"- Dog Pic: {payload.image_url}",
                sentence=f"Dog pic: {payload.image_url}",
                required_evidence=(payload.image_url,),
            )
        ]
    return _empty_facts()


def _trivia_facts(args: ToolArgs | dict[str, Any], payload: Any) -> list[GroundedFact]:
    if isinstance(payload, ToolError):
        detail = payload.details or payload.error
        return [
            GroundedFact(
                label="Trivia",
                display_text=f"- Trivia: unavailable ({detail})",
                required_evidence=("unavailable", detail),
                status="failed",
            )
        ]
    if isinstance(payload, TriviaResult):
        choices = payload.incorrect_answers + [payload.correct_answer]
        rendered_choices = ", ".join(choices)
        display = f"- Trivia: {payload.question} Choices: {rendered_choices}"
        return [
            GroundedFact(
                label="Trivia",
                display_text=display,
                sentence=f"Trivia: {payload.question} Choices: {rendered_choices}.",
                required_evidence=(payload.question,),
            )
        ]
    return _empty_facts()


GROUNDING_RENDERERS = {
    "city_to_coords": _city_lookup_facts,
    "get_weather": _weather_facts,
    "book_recs": _book_facts,
    "random_joke": _joke_facts,
    "random_dog": _dog_facts,
    "trivia": _trivia_facts,
}


def _validate_grounding_renderers() -> None:
    for tool_name, renderer in GROUNDING_RENDERERS.items():
        if not tool_name:
            raise RuntimeError("Grounding renderer registry contains an empty tool name")
        if not callable(renderer):
            raise RuntimeError(f"Grounding renderer for {tool_name} must be callable")


_validate_grounding_renderers()


def extract_grounded_facts(tool_name: str, args: ToolArgs | dict[str, Any], payload: Any) -> list[GroundedFact]:
    """Extract grounded facts from one parsed execution step."""
    renderer = GROUNDING_RENDERERS.get(tool_name)
    if renderer is None:
        return _empty_facts()
    return renderer(args, payload)


def build_grounded_facts(steps: Iterable[Any]) -> list[GroundedFact]:
    """Build grounded facts from execution-step state."""
    facts: list[GroundedFact] = []
    for step in steps:
        if getattr(step, "kind", "") not in {"tool_call", "tool_invalid"}:
            continue
        tool_name, args, payload = _step_fields(step)
        facts.extend(extract_grounded_facts(tool_name, args, payload))
    return facts


def build_grounded_items(user_prompt: str, steps: Iterable[Any]) -> List[Any]:
    """Return simple objects with title/detail from execution steps."""
    items: List[Any] = []
    lowered = user_prompt.lower()

    for fact in build_grounded_facts(steps):
        if not fact.display_text:
            continue
        title, detail = fact.display_text[2:].split(": ", 1)
        items.append(type("GroundedCompat", (), {"title": title, "detail": detail})())

    if not items and "weekend" in lowered:
        items.append(type("GroundedCompat", (), {"title": "Detail", "detail": "Try a cozy cafe stop, a short walk, and a relaxing book session this weekend."})())

    return items


def render_compact_step_summaries(
    user_prompt: str,
    steps: List[Any],
) -> List[str]:
    """Render compact grounded step summaries from execution-step state."""
    rendered = [fact.display_text for fact in build_grounded_facts(steps)]
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

    facts = build_grounded_facts(tool_steps)
    detail_lines = [fact.display_text for fact in facts]
    fact_lines = [fact.sentence for fact in facts if fact.sentence]

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
