from __future__ import annotations

"""Central tool behavior metadata for orchestrator control logic."""

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from pydantic import TypeAdapter

from agent.policies.guardrails import infer_book_limit, infer_book_topic, infer_city
from schemas.tools import (
    BookArgs,
    BookResults,
    CityArgs,
    DogResult,
    EmptyArgs,
    GeoResult,
    JokeResult,
    ToolError,
    ToolArgs,
    TriviaResult,
    WeatherArgs,
    WeatherResult,
)


ArgPolicy = Literal["city", "weather_coords", "book_recs", "empty", "passthrough"]


@dataclass(frozen=True)
class ToolBehaviorSpec:
    """Declared behavior contract for one tool in orchestrator control flow."""

    name: str
    category_label: str
    purpose: str
    input_contract: str
    prompt_example: str
    markers: tuple[str, ...]
    fulfills_requested_work: bool
    arg_policy: ArgPolicy
    payload_adapter: TypeAdapter[Any]

    def parse_payload(self, payload: Any) -> Any:
        """Parse this tool's raw payload into its declared result contract."""
        if not isinstance(payload, dict):
            return payload

        if "error" in payload:
            try:
                return ToolError.model_validate(payload)
            except Exception:
                return payload

        try:
            return self.payload_adapter.validate_python(payload)
        except Exception:
            return payload

    def normalize_args(
        self,
        args: dict[str, Any],
        *,
        user_prompt: str,
        request_analysis: Any,
    ) -> tuple[ToolArgs | None, str | None]:
        """Normalize ReAct-produced args into this tool's input contract."""
        args = dict(args or {})

        if self.arg_policy == "city":
            city = (
                args.get("city")
                or (request_analysis.city if request_analysis is not None else None)
                or infer_city(user_prompt)
            )
            if not city:
                return None, "city is required"
            return CityArgs(city=str(city)), None

        if self.arg_policy == "weather_coords":
            latitude = args.get("latitude")
            longitude = args.get("longitude")
            if latitude is None or longitude is None:
                return None, "latitude and longitude are required"
            try:
                return WeatherArgs(latitude=float(latitude), longitude=float(longitude)), None
            except (TypeError, ValueError):
                return None, "latitude and longitude must be numeric"

        if self.arg_policy == "book_recs":
            topic = (
                args.get("topic")
                or args.get("param")
                or (request_analysis.book_topic if request_analysis is not None else None)
                or infer_book_topic(user_prompt)
            )
            limit = (
                (request_analysis.book_limit if request_analysis is not None else None)
                or args.get("limit")
                or infer_book_limit(user_prompt)
            )
            if not topic:
                return None, "topic is required"
            try:
                safe_limit = max(1, min(int(limit), 10))
            except (TypeError, ValueError):
                safe_limit = 3
            return BookArgs(topic=str(topic), limit=safe_limit), None

        if self.arg_policy == "empty":
            return EmptyArgs(), None

        return EmptyArgs(), None


TOOL_SPECS: Mapping[str, ToolBehaviorSpec] = {
    "city_to_coords": ToolBehaviorSpec(
        name="city_to_coords",
        category_label="city lookup",
        purpose="Resolve a user-provided city name to coordinates.",
        input_contract="city:string required",
        prompt_example='city_to_coords args={"city":"New York"}',
        markers=("city lookup",),
        fulfills_requested_work=False,
        arg_policy="city",
        payload_adapter=TypeAdapter(GeoResult | ToolError),
    ),
    "get_weather": ToolBehaviorSpec(
        name="get_weather",
        category_label="weather",
        purpose="Fetch current weather for known coordinates.",
        input_contract="latitude:number required, longitude:number required",
        prompt_example='get_weather args={"latitude":40.7128,"longitude":-74.0060}',
        markers=("weather",),
        fulfills_requested_work=True,
        arg_policy="weather_coords",
        payload_adapter=TypeAdapter(WeatherResult | ToolError),
    ),
    "book_recs": ToolBehaviorSpec(
        name="book_recs",
        category_label="books",
        purpose="Fetch book recommendations for a requested topic.",
        input_contract="topic:string required, limit:integer optional",
        prompt_example='book_recs args={"topic":"mystery","limit":3}',
        markers=("books", "book", "book ideas", "mystery book"),
        fulfills_requested_work=True,
        arg_policy="book_recs",
        payload_adapter=TypeAdapter(BookResults | ToolError),
    ),
    "random_joke": ToolBehaviorSpec(
        name="random_joke",
        category_label="joke",
        purpose="Fetch one safe random joke.",
        input_contract="no arguments",
        prompt_example="random_joke args={}",
        markers=("joke",),
        fulfills_requested_work=True,
        arg_policy="empty",
        payload_adapter=TypeAdapter(JokeResult | ToolError),
    ),
    "random_dog": ToolBehaviorSpec(
        name="random_dog",
        category_label="dog pic",
        purpose="Fetch one random dog image URL.",
        input_contract="no arguments",
        prompt_example="random_dog args={}",
        markers=("dog pic", "dog photo", "dog", "pawsome pic"),
        fulfills_requested_work=True,
        arg_policy="empty",
        payload_adapter=TypeAdapter(DogResult | ToolError),
    ),
    "trivia": ToolBehaviorSpec(
        name="trivia",
        category_label="trivia",
        purpose="Fetch one multiple-choice trivia question with its answer.",
        input_contract="no arguments",
        prompt_example="trivia args={}",
        markers=("trivia",),
        fulfills_requested_work=True,
        arg_policy="empty",
        payload_adapter=TypeAdapter(TriviaResult | ToolError),
    ),
}


def _validate_tool_specs() -> None:
    for name, spec in TOOL_SPECS.items():
        if spec.name != name:
            raise RuntimeError(f"Tool spec key/name mismatch for {name}")
        if not spec.category_label.strip():
            raise RuntimeError(f"Tool spec {name} must define a category label")
        if not spec.purpose.strip():
            raise RuntimeError(f"Tool spec {name} must define a purpose")
        if not spec.input_contract.strip():
            raise RuntimeError(f"Tool spec {name} must define an input contract")
        if not spec.prompt_example.strip():
            raise RuntimeError(f"Tool spec {name} must define a prompt example")
        if not spec.markers:
            raise RuntimeError(f"Tool spec {name} must define at least one marker")
        if any(not marker.strip() for marker in spec.markers):
            raise RuntimeError(f"Tool spec {name} has an empty marker")
        if spec.payload_adapter is None:
            raise RuntimeError(f"Tool spec {name} must define a payload adapter")


_validate_tool_specs()


def get_tool_spec(tool_name: str) -> ToolBehaviorSpec:
    """Return validated tool behavior metadata for one known tool."""
    try:
        return TOOL_SPECS[tool_name]
    except KeyError as exc:
        raise ValueError(f"Unknown tool spec: {tool_name}") from exc
