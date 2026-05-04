from __future__ import annotations

"""Request-analysis and planning helper functions for Weekend Wizard."""

from dataclasses import dataclass
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from schemas.tools import BookResults, DogResult, ToolError, WeatherResult


_TOOL_ORDER = (
    "get_weather",
    "book_recs",
    "random_joke",
    "random_dog",
    "trivia",
)

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

_BOOK_TOPIC_STOPWORDS = {
    "a",
    "an",
    "and",
    "book",
    "books",
    "idea",
    "ideas",
    "me",
    "some",
    "the",
}

_WEATHER_KEYWORDS = (
    "weather",
    "temperature",
    "forecast",
    "rain",
    "raining",
    "sunny",
    "cloudy",
    "snow",
    "snowing",
    "wind",
    "windy",
    "humid",
    "humidity",
    "hot",
    "cold",
)


@dataclass(frozen=True)
class RequestAnalysis:
    """Deterministic request analysis for a supported Weekend Wizard prompt.

    Attributes:
        requested_tools: Ordered tool categories inferred from the user request.
        coords: Optional coordinates parsed directly from the prompt.
        city: Optional city name inferred from the prompt.
        book_topic: Normalized topic used for book recommendations.
        book_limit: Requested number of book recommendations.
    """

    requested_tools: Tuple[str, ...]
    coords: Optional[Tuple[float, float]] = None
    city: Optional[str] = None
    book_topic: str = "books"
    book_limit: int = 3

    @property
    def needs_city_lookup(self) -> bool:
        return self.coords is None and self.city is not None


@dataclass(frozen=True)
class FinalAnswerValidation:
    """Deterministic validation result for a reflected final answer."""

    is_valid: bool
    missing_tools: Tuple[str, ...] = ()


def parse_coords(text: str) -> Optional[Tuple[float, float]]:
    """Parse latitude and longitude coordinates from free-form text."""
    patterns = (
        r"\((-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\)",
        r"\b(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        latitude = float(match.group(1))
        longitude = float(match.group(2))
        if -90 <= latitude <= 90 and -180 <= longitude <= 180:
            return latitude, longitude
    return None


def infer_city(text: str) -> Optional[str]:
    """Infer a city phrase from a prompt when explicit coordinates are absent."""
    patterns = [
        r"\bin ([A-Z][a-zA-Z]+(?:[ \-][A-Z][a-zA-Z]+)*)",
        r"\bfor ([A-Z][a-zA-Z]+(?:[ \-][A-Z][a-zA-Z]+)*)",
        r"\bat ([A-Z][a-zA-Z]+(?:[ \-][A-Z][a-zA-Z]+)*)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            if "(" not in candidate and ")" not in candidate:
                return candidate

    return None


def requested_tools(prompt: str) -> Set[str]:
    """Infer which tool categories the user is explicitly asking for."""
    lowered = prompt.lower()
    requested: Set[str] = set()

    if any(keyword in lowered for keyword in _WEATHER_KEYWORDS):
        requested.add("get_weather")
    if "book" in lowered or "read" in lowered or "theme" in lowered:
        requested.add("book_recs")
    if "joke" in lowered:
        requested.add("random_joke")
    if "dog" in lowered:
        requested.add("random_dog")
    if "trivia" in lowered:
        requested.add("trivia")

    return requested


def infer_book_limit(prompt: str) -> int:
    """Infer how many book recommendations the user wants."""
    digit_match = re.search(r"\b(\d+)\s+(?:\w+\s+){0,2}book", prompt, flags=re.IGNORECASE)
    if digit_match:
        return max(1, min(int(digit_match.group(1)), 10))

    lowered = prompt.lower()
    for word, value in _NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+(?:\w+\s+){{0,2}}book", lowered):
            return value

    return 3


def infer_book_topic(prompt: str) -> str:
    """Infer a reasonable book-search topic from the user's prompt."""
    lowered = prompt.lower()

    if "cozy mystery" in lowered:
        return "cozy mystery"
    if "mystery" in lowered:
        return "mystery"
    if "fantasy" in lowered:
        return "fantasy"
    if "romance" in lowered:
        return "romance"
    if "thriller" in lowered:
        return "thriller"
    if "sci-fi" in lowered or "science fiction" in lowered:
        return "science fiction"

    patterns = (
        r"\bbook(?:s| ideas?)\s+(?:about|on|for)\s+([a-z][a-z\s\-]+)",
        r"\b(?:\d+\s+)?([a-z][a-z\s\-]+?)\s+book(?:s| ideas?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        tokens = [
            token
            for token in re.split(r"[\s\-]+", match.group(1).strip())
            if token and token not in _BOOK_TOPIC_STOPWORDS and token not in _NUMBER_WORDS
        ]
        if tokens:
            return " ".join(tokens[:3])

    return "books"


def analyze_request(
    prompt: str,
    available_tools: Iterable[str],
) -> Optional[RequestAnalysis]:
    """Build a deterministic request analysis when the prompt fits the common flow."""
    requested = requested_tools(prompt)
    if not requested:
        return None

    available = set(available_tools)
    if not requested.issubset(available):
        return None

    coords = parse_coords(prompt)
    city = infer_city(prompt)

    if "get_weather" in requested and coords is None and city is None:
        return None
    if "get_weather" in requested and coords is None and city is not None and "city_to_coords" not in available:
        return None

    ordered_tools = tuple(tool_name for tool_name in _TOOL_ORDER if tool_name in requested)
    return RequestAnalysis(
        requested_tools=ordered_tools,
        coords=coords,
        city=city,
        book_topic=infer_book_topic(prompt),
        book_limit=infer_book_limit(prompt),
    )


def missing_requested_tools(prompt: str, payloads: Dict[str, Any]) -> List[str]:
    """Return requested tools that have not yet produced payloads."""
    return [
        tool_name
        for tool_name in requested_tools(prompt)
        if tool_name not in payloads
    ]


def validate_final_answer(prompt: str, answer: str, payloads: Dict[str, Any]) -> FinalAnswerValidation:
    """Validate that a reflected answer still covers required observed content."""
    missing: List[str] = []

    for tool_name in requested_tools(prompt):
        payload = payloads.get(tool_name)
        if payload is None or isinstance(payload, ToolError):
            continue
        if not _answer_covers_tool_output(tool_name, answer, payload):
            missing.append(tool_name)

    return FinalAnswerValidation(is_valid=not missing, missing_tools=tuple(missing))


def _answer_covers_tool_output(tool_name: str, answer: str, payload: Any) -> bool:
    lowered_answer = answer.lower()

    if tool_name == "get_weather" and isinstance(payload, WeatherResult):
        return _answer_covers_weather(lowered_answer, payload)
    if tool_name == "book_recs" and isinstance(payload, BookResults):
        return all(
            not book.title or book.title.lower() in lowered_answer
            for book in payload.results
        )
    if tool_name == "random_dog" and isinstance(payload, DogResult):
        return payload.image_url.lower() in lowered_answer

    return True


def _answer_covers_weather(answer: str, payload: WeatherResult) -> bool:
    has_temperature = False
    if payload.temperature is not None:
        temperature_text = format(payload.temperature, "g")
        has_temperature = temperature_text in answer

    has_summary = False
    if payload.weather_summary:
        summary_tokens = [
            token
            for token in re.findall(r"[a-z]+", payload.weather_summary.lower())
            if token not in {"and", "the"}
        ]
        has_summary = all(
            token in answer or (token == "sky" and "skies" in answer)
            for token in summary_tokens
        )

    return has_temperature or has_summary
