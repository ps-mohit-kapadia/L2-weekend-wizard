from __future__ import annotations

from typing import Any, Dict

import requests

from mcp_runtime.registry import mcp
from tools.shared import error_payload, get_json


WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


@mcp.tool()
async def get_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Current weather for coordinates via Open-Meteo."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,weather_code,wind_speed_10m",
        "timezone": "auto",
    }

    try:
        data = await get_json("https://api.open-meteo.com/v1/forecast", params=params)
    except requests.RequestException as exc:
        return error_payload("weather", exc)

    current = data.get("current", {})
    units = data.get("current_units", {})
    weather_code = current.get("weather_code")

    return {
        "latitude": latitude,
        "longitude": longitude,
        "observed_at": current.get("time"),
        "temperature": current.get("temperature_2m"),
        "temperature_unit": units.get("temperature_2m", "C"),
        "wind_speed": current.get("wind_speed_10m"),
        "wind_speed_unit": units.get("wind_speed_10m", "km/h"),
        "weather_code": weather_code,
        "weather_summary": WEATHER_CODES.get(weather_code, "unknown conditions"),
    }
