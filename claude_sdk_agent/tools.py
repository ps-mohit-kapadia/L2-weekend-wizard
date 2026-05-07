from __future__ import annotations

"""Structured Claude Agent SDK tool wrappers for Weekend Wizard."""

import json
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool
from claude_sdk_agent.config import ClaudeSdkAgentConfig
from tools.books import book_recs as l2_book_recs
from tools.entertainment import random_dog as l2_random_dog
from tools.entertainment import random_joke as l2_random_joke
from tools.entertainment import trivia as l2_trivia
from tools.geo import city_to_coords as l2_city_to_coords
from tools.weather import get_weather as l2_get_weather


def _tool_result_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a Weekend Wizard payload into a Claude SDK MCP tool result."""
    text = json.dumps(payload)
    is_error = isinstance(payload, dict) and "error" in payload
    return {
        "content": [
            {
                "type": "text",
                "text": text,
            }
        ],
        "is_error": is_error,
    }


@tool(
    "random_joke",
    "Return one safe one-line joke for the user.",
    {},
)
async def random_joke_tool(_args: dict[str, Any]) -> dict[str, Any]:
    """Wrap the existing L2 joke behavior as an in-process SDK MCP tool."""
    payload = l2_random_joke()
    return _tool_result_from_payload(payload)


@tool(
    "random_dog",
    "Return one random dog image URL for the user.",
    {},
)
async def random_dog_tool(_args: dict[str, Any]) -> dict[str, Any]:
    """Wrap the existing L2 dog behavior as an in-process SDK MCP tool."""
    payload = l2_random_dog()
    return _tool_result_from_payload(payload)


@tool(
    "trivia",
    "Return one multiple-choice trivia question.",
    {},
)
async def trivia_tool(_args: dict[str, Any]) -> dict[str, Any]:
    """Wrap the existing L2 trivia behavior as an in-process SDK MCP tool."""
    payload = l2_trivia()
    return _tool_result_from_payload(payload)


@tool(
    "book_recs",
    "Return book recommendations for a topic and limit.",
    {"topic": str, "limit": int},
)
async def book_recs_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Wrap the existing L2 book behavior as an in-process SDK MCP tool."""
    payload = l2_book_recs(topic=args["topic"], limit=args["limit"])
    return _tool_result_from_payload(payload)


@tool(
    "city_to_coords",
    "Resolve a city name to coordinates.",
    {"city": str},
)
async def city_to_coords_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Wrap the existing L2 geocoding behavior as an in-process SDK MCP tool."""
    payload = l2_city_to_coords(city=args["city"])
    return _tool_result_from_payload(payload)


@tool(
    "get_weather",
    "Return current weather for latitude and longitude.",
    {"latitude": float, "longitude": float},
)
async def get_weather_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Wrap the existing L2 weather behavior as an in-process SDK MCP tool."""
    payload = l2_get_weather(latitude=args["latitude"], longitude=args["longitude"])
    return _tool_result_from_payload(payload)


def build_sdk_server(config: ClaudeSdkAgentConfig):
    """Build the in-process SDK MCP server for the current slice."""
    return create_sdk_mcp_server(
        name=config.server_name,
        version=config.server_version,
        tools=[
            random_joke_tool,
            random_dog_tool,
            trivia_tool,
            book_recs_tool,
            city_to_coords_tool,
            get_weather_tool,
        ],
    )
