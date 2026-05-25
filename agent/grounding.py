from __future__ import annotations

"""Grounding helpers for tool-result parsing and final answer composition."""

from dataclasses import dataclass
import json
from typing import Any, List, Optional

from schemas.agent import ToolObservation
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


@dataclass(frozen=True)
class GroundedItem:
    """Normalized grounded content derived from one tool result."""

    title: str
    detail: str
    fact: Optional[str] = None


@dataclass(frozen=True)
class ParsedObservation:
    """Parsed tool observation that preserves execution order and args context."""

    tool_name: str
    args: dict[str, Any]
    payload: Any


def parse_tool_payload_text(tool_name: str, payload_text: str) -> Any:
    """Parse one serialized tool payload into a typed payload when possible."""
    try:
        return parse_tool_payload(tool_name, json.loads(payload_text))
    except json.JSONDecodeError:
        return payload_text


def parse_tool_observations(tool_observations: List[ToolObservation]) -> List[ParsedObservation]:
    """Parse structured tool observations while preserving multiplicity and order."""
    parsed: List[ParsedObservation] = []
    for observation in tool_observations:
        parsed.append(
            ParsedObservation(
                tool_name=observation.tool_name,
                args=dict(observation.args),
                payload=parse_tool_payload_text(observation.tool_name, observation.payload),
            )
        )
    return parsed


def _coords_label(args: dict[str, Any]) -> str:
    latitude = args.get("latitude")
    longitude = args.get("longitude")
    if latitude is None or longitude is None:
        return "requested location"
    return f"{latitude}, {longitude}"


def build_grounded_items(user_prompt: str, observations: List[ParsedObservation]) -> List[GroundedItem]:
    """Build normalized grounded items from parsed tool observations."""
    items: List[GroundedItem] = []
    lowered = user_prompt.lower()

    for observation in observations:
        payload = observation.payload

        if observation.tool_name == "city_to_coords":
            if isinstance(payload, ToolError):
                items.append(
                    GroundedItem(
                        title="City Lookup",
                        detail=f"unavailable ({payload.details or payload.error})",
                    )
                )
            elif isinstance(payload, GeoResult):
                items.append(
                    GroundedItem(
                        title="City Lookup",
                        detail=f"{payload.city}: {payload.latitude}, {payload.longitude}",
                        fact=f"Resolved {payload.city} to {payload.latitude}, {payload.longitude}.",
                    )
                )
            continue

        if observation.tool_name == "get_weather":
            if isinstance(payload, ToolError):
                items.append(
                    GroundedItem(
                        title="Weather",
                        detail=f"{_coords_label(observation.args)} unavailable ({payload.details or payload.error})",
                    )
                )
            elif isinstance(payload, WeatherResult) and payload.temperature is not None:
                detail = (
                    f"{_coords_label(observation.args)}: "
                    f"{payload.temperature}{payload.temperature_unit or ''}, "
                    f"{payload.weather_summary or 'current conditions'}"
                )
                items.append(
                    GroundedItem(
                        title="Weather",
                        detail=detail,
                        fact=(
                            f"Weather for {_coords_label(observation.args)}: "
                            f"{payload.temperature}{payload.temperature_unit or ''}, "
                            f"{payload.weather_summary or 'current conditions'}."
                        ),
                    )
                )
            continue

        if observation.tool_name == "book_recs":
            if isinstance(payload, ToolError):
                items.append(
                    GroundedItem(
                        title="Books",
                        detail=f"unavailable ({payload.details or payload.error})",
                    )
                )
            elif isinstance(payload, BookResults) and payload.results:
                titles = [
                    f"{book.title} by {book.author}"
                    for book in payload.results[:2]
                    if book.title
                ]
                if titles:
                    detail = "; ".join(titles)
                    items.append(
                        GroundedItem(
                            title="Books",
                            detail=detail,
                            fact=f"Book ideas for {payload.topic}: {detail}.",
                        )
                    )
            continue

        if observation.tool_name == "random_joke":
            if isinstance(payload, ToolError):
                items.append(
                    GroundedItem(
                        title="Joke",
                        detail=f"unavailable ({payload.details or payload.error})",
                    )
                )
            elif isinstance(payload, JokeResult):
                items.append(
                    GroundedItem(
                        title="Joke",
                        detail=payload.joke,
                        fact=f"Joke: {payload.joke}",
                    )
                )
            continue

        if observation.tool_name == "random_dog":
            if isinstance(payload, ToolError):
                items.append(
                    GroundedItem(
                        title="Dog Pic",
                        detail=f"unavailable ({payload.details or payload.error})",
                    )
                )
            elif isinstance(payload, DogResult):
                items.append(
                    GroundedItem(
                        title="Dog Pic",
                        detail=payload.image_url,
                        fact=f"Dog pic: {payload.image_url}",
                    )
                )
            continue

        if observation.tool_name == "trivia":
            if isinstance(payload, ToolError):
                items.append(
                    GroundedItem(
                        title="Trivia",
                        detail=f"unavailable ({payload.details or payload.error})",
                    )
                )
            elif isinstance(payload, TriviaResult):
                choices = payload.incorrect_answers + [payload.correct_answer]
                detail = f"{payload.question} Choices: {', '.join(choices)}"
                items.append(
                    GroundedItem(
                        title="Trivia",
                        detail=detail,
                        fact=f"Trivia: {detail}.",
                    )
                )
            continue

    if not items and "weekend" in lowered:
        fallback = "Try a cozy cafe stop, a short walk, and a relaxing book session this weekend."
        items.append(
            GroundedItem(
                title="Detail",
                detail=fallback,
                fact=fallback,
            )
        )

    return items


def render_grounded_sections(items: List[GroundedItem]) -> List[str]:
    """Render normalized grounded items into answer sections."""
    return [f"- {item.title}: {item.detail}" for item in items]


def render_compact_observation_summaries(
    user_prompt: str,
    tool_observations: List[ToolObservation],
) -> List[str]:
    """Render compact grounded observation summaries for prompts."""
    grounded_items = build_grounded_items(user_prompt, parse_tool_observations(tool_observations))
    rendered: List[str] = []
    for item in grounded_items:
        detail = item.detail
        if item.title == "Dog Pic":
            detail = "fetched one dog image"
        rendered.append(f"- {item.title}: {detail}")
    return rendered


def compose_grounded_answer_from_payloads(
    user_prompt: str,
    answer: str,
    observations: List[ParsedObservation],
) -> str:
    """Compose the final grounded answer from parsed tool observations."""
    if not observations:
        return answer

    grounded_items = build_grounded_items(user_prompt, observations)
    if not grounded_items:
        return answer

    lowered = user_prompt.lower()
    is_plan_request = any(word in lowered for word in ("plan", "weekend", "saturday", "sunday"))
    grounded_facts = [item.fact for item in grounded_items if item.fact]

    if len(observations) > 1 or is_plan_request:
        intro = "Weekend Wizard Plan" if is_plan_request else "Weekend Wizard Results"
        outro = []
        if is_plan_request:
            outro.append("Enjoy the vibe and follow the links if something catches your eye.")
        body = render_grounded_sections(grounded_items)
        return "\n".join([intro, *body, *outro])

    if len(grounded_facts) == 1:
        return grounded_facts[0]

    if grounded_facts:
        return " ".join(grounded_facts)

    return answer


def compose_grounded_answer_from_observations(
    user_prompt: str,
    answer: str,
    tool_observations: List[ToolObservation],
) -> str:
    """Compose the final grounded answer from structured tool observations."""
    return compose_grounded_answer_from_payloads(
        user_prompt,
        answer,
        parse_tool_observations(tool_observations),
    )
