from __future__ import annotations

"""Grounding helpers for tool-result parsing and final answer composition."""

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from schemas.agent import ToolObservation
from schemas.tools import (
    BookResults,
    DogResult,
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


def parse_tool_payload_text(tool_name: str, payload_text: str) -> Any:
    """Parse one serialized tool payload into a typed payload when possible."""
    try:
        return parse_tool_payload(tool_name, json.loads(payload_text))
    except json.JSONDecodeError:
        return payload_text


def parse_tool_observations(tool_observations: List[ToolObservation]) -> Dict[str, Any]:
    """Parse structured tool observations into typed payloads keyed by tool name."""
    payloads: Dict[str, Any] = {}
    for observation in tool_observations:
        payloads[observation.tool_name] = parse_tool_payload_text(
            observation.tool_name,
            observation.payload,
        )
    return payloads


def build_grounded_items(user_prompt: str, payloads: Dict[str, Any]) -> List[GroundedItem]:
    """Build normalized grounded items from typed tool payloads.

    Args:
        user_prompt: The original user request.
        payloads: Parsed tool payloads keyed by tool name.

    Returns:
        Normalized grounded items for response rendering.
    """
    items: List[GroundedItem] = []
    lowered = user_prompt.lower()

    weather = payloads.get("get_weather")
    if isinstance(weather, ToolError):
        items.append(
            GroundedItem(
                title="Weather",
                detail=f"unavailable ({weather.details or weather.error})",
            )
        )
    elif isinstance(weather, WeatherResult) and weather.temperature is not None:
        detail = f"{weather.temperature}{weather.temperature_unit or ''}, {weather.weather_summary or 'current conditions'}"
        items.append(
            GroundedItem(
                title="Weather",
                detail=detail,
                fact=f"Weather now: {detail}.",
            )
        )

    books = payloads.get("book_recs")
    if isinstance(books, ToolError):
        items.append(
            GroundedItem(
                title="Books",
                detail=f"unavailable ({books.details or books.error})",
            )
        )
    elif isinstance(books, BookResults) and books.results:
        titles = [
            f"{book.title} by {book.author}"
            for book in books.results[:2]
            if book.title
        ]
        if titles:
            detail = "; ".join(titles)
            items.append(
                GroundedItem(
                    title="Books",
                    detail=detail,
                    fact=f"Book ideas for {books.topic}: {detail}.",
                )
            )

    joke = payloads.get("random_joke")
    if isinstance(joke, ToolError):
        items.append(
            GroundedItem(
                title="Joke",
                detail=f"unavailable ({joke.details or joke.error})",
            )
        )
    elif isinstance(joke, JokeResult):
        items.append(
            GroundedItem(
                title="Joke",
                detail=joke.joke,
                fact=f"Joke: {joke.joke}",
            )
        )

    dog = payloads.get("random_dog")
    if isinstance(dog, ToolError):
        items.append(
            GroundedItem(
                title="Dog Pic",
                detail=f"unavailable ({dog.details or dog.error})",
            )
        )
    elif isinstance(dog, DogResult):
        items.append(
            GroundedItem(
                title="Dog Pic",
                detail=dog.image_url,
                fact=f"Dog pic: {dog.image_url}",
            )
        )

    trivia = payloads.get("trivia")
    if isinstance(trivia, ToolError):
        items.append(
            GroundedItem(
                title="Trivia",
                detail=f"unavailable ({trivia.details or trivia.error})",
            )
        )
    elif isinstance(trivia, TriviaResult):
        choices = trivia.incorrect_answers + [trivia.correct_answer]
        detail = f"{trivia.question} Choices: {', '.join(choices)}"
        items.append(
            GroundedItem(
                title="Trivia",
                detail=detail,
                fact=f"Trivia: {detail}.",
            )
        )

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
    """Render normalized grounded items into answer sections.

    Args:
        items: Normalized grounded items.

    Returns:
        Formatted answer sections suitable for multi-tool responses.
    """
    return [f"- {item.title}: {item.detail}" for item in items]


def compose_grounded_answer_from_payloads(
    user_prompt: str,
    answer: str,
    payloads: Dict[str, Any],
) -> str:
    """Compose the final grounded answer from parsed tool payloads."""
    if not payloads:
        return answer

    grounded_items = build_grounded_items(user_prompt, payloads)
    if not grounded_items:
        return answer

    lowered = user_prompt.lower()
    is_plan_request = any(word in lowered for word in ("plan", "weekend", "saturday", "sunday"))
    grounded_facts = [item.fact for item in grounded_items if item.fact]

    if len(payloads) > 1 or is_plan_request:
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
