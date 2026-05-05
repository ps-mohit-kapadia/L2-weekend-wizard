from __future__ import annotations

"""Grounding helpers for tool-result parsing and grounded draft composition."""

from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, List, Optional

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


ToolRenderer = Callable[[Any], Optional[GroundedItem]]


def parse_tool_payload_text(tool_name: str, payload_text: str) -> Any:
    """Parse one serialized tool payload into a typed payload when possible."""
    try:
        return parse_tool_payload(tool_name, json.loads(payload_text))
    except json.JSONDecodeError:
        return payload_text


def parse_tool_payloads(payload_texts: Dict[str, str]) -> Dict[str, Any]:
    """Parse serialized tool payloads once and return them keyed by tool name."""
    return {
        tool_name: parse_tool_payload_text(tool_name, payload_text)
        for tool_name, payload_text in payload_texts.items()
    }


def _render_weather_item(payload: Any) -> Optional[GroundedItem]:
    if isinstance(payload, ToolError):
        return GroundedItem(
            title="Weather",
            detail=f"unavailable ({payload.details or payload.error})",
        )
    if isinstance(payload, WeatherResult) and payload.temperature is not None:
        detail = f"{payload.temperature}{payload.temperature_unit or ''}, {payload.weather_summary or 'current conditions'}"
        return GroundedItem(
            title="Weather",
            detail=detail,
            fact=f"Weather now: {detail}.",
        )
    return None


def _render_books_item(payload: Any) -> Optional[GroundedItem]:
    if isinstance(payload, ToolError):
        return GroundedItem(
            title="Books",
            detail=f"unavailable ({payload.details or payload.error})",
        )
    if isinstance(payload, BookResults) and payload.results:
        titles = [
            f"{book.title} by {book.author}"
            for book in payload.results
            if book.title
        ]
        if titles:
            detail = "; ".join(titles)
            return GroundedItem(
                title="Books",
                detail=detail,
                fact=f"Book ideas for {payload.topic}: {detail}.",
            )
    return None


def _render_joke_item(payload: Any) -> Optional[GroundedItem]:
    if isinstance(payload, ToolError):
        return GroundedItem(
            title="Joke",
            detail=f"unavailable ({payload.details or payload.error})",
        )
    if isinstance(payload, JokeResult):
        return GroundedItem(
            title="Joke",
            detail=payload.joke,
            fact=f"Joke: {payload.joke}",
        )
    return None


def _render_dog_item(payload: Any) -> Optional[GroundedItem]:
    if isinstance(payload, ToolError):
        return GroundedItem(
            title="Dog Pic",
            detail=f"unavailable ({payload.details or payload.error})",
        )
    if isinstance(payload, DogResult):
        return GroundedItem(
            title="Dog Pic",
            detail=payload.image_url,
            fact=f"Dog pic: {payload.image_url}",
        )
    return None


def _render_trivia_item(payload: Any) -> Optional[GroundedItem]:
    if isinstance(payload, ToolError):
        return GroundedItem(
            title="Trivia",
            detail=f"unavailable ({payload.details or payload.error})",
        )
    if isinstance(payload, TriviaResult):
        choices = payload.incorrect_answers + [payload.correct_answer]
        detail = f"{payload.question} Choices: {', '.join(choices)}"
        return GroundedItem(
            title="Trivia",
            detail=detail,
            fact=f"Trivia: {detail}.",
        )
    return None


_TOOL_RENDERERS: Dict[str, ToolRenderer] = {
    "get_weather": _render_weather_item,
    "book_recs": _render_books_item,
    "random_joke": _render_joke_item,
    "random_dog": _render_dog_item,
    "trivia": _render_trivia_item,
}

_RENDER_ORDER = (
    "get_weather",
    "book_recs",
    "random_joke",
    "random_dog",
    "trivia",
)


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

    for tool_name in _RENDER_ORDER:
        payload = payloads.get(tool_name)
        if payload is None:
            continue
        item = _TOOL_RENDERERS[tool_name](payload)
        if item is not None:
            items.append(item)

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


def build_grounded_draft_from_payloads(
    user_prompt: str,
    answer: str,
    payloads: Dict[str, Any],
) -> str:
    """Build a grounded draft answer from parsed tool payloads."""
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
