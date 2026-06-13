from __future__ import annotations

"""Central tool behavior metadata for orchestrator control logic."""

from dataclasses import dataclass
from typing import Literal, Mapping


ArgPolicy = Literal["city", "weather_coords", "book_recs", "empty", "passthrough"]


@dataclass(frozen=True)
class ToolBehaviorSpec:
    """Declared behavior contract for one tool in orchestrator control flow."""

    name: str
    category_label: str
    prompt_example: str
    markers: tuple[str, ...]
    fulfills_requested_work: bool
    arg_policy: ArgPolicy


TOOL_SPECS: Mapping[str, ToolBehaviorSpec] = {
    "city_to_coords": ToolBehaviorSpec(
        name="city_to_coords",
        category_label="city lookup",
        prompt_example='city_to_coords args={"city":"New York"}',
        markers=("city lookup",),
        fulfills_requested_work=False,
        arg_policy="city",
    ),
    "get_weather": ToolBehaviorSpec(
        name="get_weather",
        category_label="weather",
        prompt_example='get_weather args={"latitude":40.7128,"longitude":-74.0060}',
        markers=("weather",),
        fulfills_requested_work=True,
        arg_policy="weather_coords",
    ),
    "book_recs": ToolBehaviorSpec(
        name="book_recs",
        category_label="books",
        prompt_example='book_recs args={"topic":"mystery","limit":3}',
        markers=("books", "book", "book ideas", "mystery book"),
        fulfills_requested_work=True,
        arg_policy="book_recs",
    ),
    "random_joke": ToolBehaviorSpec(
        name="random_joke",
        category_label="joke",
        prompt_example="random_joke args={}",
        markers=("joke",),
        fulfills_requested_work=True,
        arg_policy="empty",
    ),
    "random_dog": ToolBehaviorSpec(
        name="random_dog",
        category_label="dog pic",
        prompt_example="random_dog args={}",
        markers=("dog pic", "dog photo", "dog", "pawsome pic"),
        fulfills_requested_work=True,
        arg_policy="empty",
    ),
    "trivia": ToolBehaviorSpec(
        name="trivia",
        category_label="trivia",
        prompt_example="trivia args={}",
        markers=("trivia",),
        fulfills_requested_work=True,
        arg_policy="empty",
    ),
}


def _validate_tool_specs() -> None:
    for name, spec in TOOL_SPECS.items():
        if spec.name != name:
            raise RuntimeError(f"Tool spec key/name mismatch for {name}")
        if not spec.category_label.strip():
            raise RuntimeError(f"Tool spec {name} must define a category label")
        if not spec.prompt_example.strip():
            raise RuntimeError(f"Tool spec {name} must define a prompt example")
        if not spec.markers:
            raise RuntimeError(f"Tool spec {name} must define at least one marker")
        if any(not marker.strip() for marker in spec.markers):
            raise RuntimeError(f"Tool spec {name} has an empty marker")


_validate_tool_specs()


def get_tool_spec(tool_name: str) -> ToolBehaviorSpec:
    """Return validated tool behavior metadata for one known tool."""
    try:
        return TOOL_SPECS[tool_name]
    except KeyError as exc:
        raise ValueError(f"Unknown tool spec: {tool_name}") from exc
