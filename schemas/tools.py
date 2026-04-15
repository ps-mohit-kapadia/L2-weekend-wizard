from __future__ import annotations

"""Typed models for MCP tool payloads used by grounding and validation."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, TypeAdapter


class ToolError(BaseModel):
    """Standard error payload returned by a tool helper.

    Attributes:
        error: Human-readable error summary.
        details: Optional low-level error details.
    """

    error: str
    details: Optional[str] = None


class GeoResult(BaseModel):
    """Geocoding payload returned by the city lookup tool.

    Attributes:
        city: Best-match city name.
        latitude: Latitude for the matched city.
        longitude: Longitude for the matched city.
        country: Optional country name.
        admin1: Optional first-level administrative region.
        timezone: Optional timezone identifier for the result.
    """

    city: str
    latitude: float
    longitude: float
    country: Optional[str] = None
    admin1: Optional[str] = None
    timezone: Optional[str] = None


class WeatherResult(BaseModel):
    """Weather payload returned by the current-weather tool.

    Attributes:
        latitude: Latitude used for the weather lookup.
        longitude: Longitude used for the weather lookup.
        observed_at: Observation timestamp from the source API.
        temperature: Current observed temperature.
        temperature_unit: Temperature unit symbol or label.
        wind_speed: Current wind speed.
        wind_speed_unit: Wind speed unit symbol or label.
        weather_code: Source weather code from Open-Meteo.
        weather_summary: Human-readable weather description.
    """

    latitude: Optional[float] = None
    longitude: Optional[float] = None
    observed_at: Optional[str] = None
    temperature: Optional[float] = None
    temperature_unit: Optional[str] = None
    wind_speed: Optional[float] = None
    wind_speed_unit: Optional[str] = None
    weather_code: Optional[int] = None
    weather_summary: Optional[str] = None


class BookItem(BaseModel):
    """One book recommendation item returned by the book tool.

    Attributes:
        title: Book title.
        author: Primary author name.
        year: First known publication year.
        work: Open Library work identifier.
    """

    title: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    work: Optional[str] = None


class BookResults(BaseModel):
    """Book search payload returned by the recommendation tool.

    Attributes:
        topic: Search topic used for the recommendation request.
        count: Optional number of returned items.
        results: Recommended books returned for the topic.
    """

    topic: str
    count: Optional[int] = None
    results: List[BookItem]


class JokeResult(BaseModel):
    """Joke payload returned by the joke tool.

    Attributes:
        joke: One-line joke text.
    """

    joke: str


class DogResult(BaseModel):
    """Dog image payload returned by the dog tool.

    Attributes:
        status: Optional API status value.
        image_url: Direct URL to the dog image.
    """

    status: Optional[str] = None
    image_url: str


class TriviaResult(BaseModel):
    """Trivia payload returned by the trivia tool.

    Attributes:
        category: Optional trivia category name.
        difficulty: Optional difficulty label.
        question: Trivia question text.
        correct_answer: Correct answer string.
        incorrect_answers: Incorrect answer options.
    """

    category: Optional[str] = None
    difficulty: Optional[str] = None
    question: str
    correct_answer: str
    incorrect_answers: List[str]


_adapters: Dict[str, TypeAdapter[Any]] = {
    "city_to_coords": TypeAdapter(GeoResult | ToolError),
    "get_weather": TypeAdapter(WeatherResult | ToolError),
    "book_recs": TypeAdapter(BookResults | ToolError),
    "random_joke": TypeAdapter(JokeResult | ToolError),
    "random_dog": TypeAdapter(DogResult | ToolError),
    "trivia": TypeAdapter(TriviaResult | ToolError),
}


def parse_tool_payload(tool_name: str, payload: Any) -> Any:
    """Parse known tool payloads into typed models when possible.

    Args:
        tool_name: Name of the tool that produced the payload.
        payload: Raw payload returned by the tool.

    Returns:
        A typed payload model for known tool shapes, or the original payload when it
        cannot be validated safely.
    """
    if not isinstance(payload, dict):
        return payload

    adapter = _adapters.get(tool_name)
    if adapter is None:
        return payload

    try:
        return adapter.validate_python(payload)
    except Exception:
        return payload
