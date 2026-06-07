from __future__ import annotations

from typing import Any

import requests

from mcp_runtime.registry import mcp
from schemas.tools import GeoResult, ToolError
from tools.shared import error_payload, get_json


@mcp.tool()
def city_to_coords(city: str) -> GeoResult | ToolError:
    """Resolve a city name to coordinates via Open-Meteo geocoding."""
    try:
        data = get_json(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
        )
    except requests.RequestException as exc:
        return error_payload("geocoding", exc)

    results = data.get("results", [])
    if not results:
        return ToolError(
            error="geocoding request failed",
            details=f"no match found for {city}",
        )

    match = results[0]
    return GeoResult(
        city=match.get("name", city),
        latitude=match.get("latitude"),
        longitude=match.get("longitude"),
        country=match.get("country"),
        admin1=match.get("admin1"),
        timezone=match.get("timezone"),
    )
