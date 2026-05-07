from __future__ import annotations

"""Structured Claude Agent SDK tool wrappers for Weekend Wizard."""

import json
from typing import Any

from claude_agent_sdk import create_sdk_mcp_server, tool
from claude_sdk_agent.config import ClaudeSdkAgentConfig
from tools.entertainment import random_joke as l2_random_joke


@tool(
    "random_joke",
    "Return one safe one-line joke for the user.",
    {},
)
async def random_joke_tool(_args: dict[str, Any]) -> dict[str, Any]:
    """Wrap the existing L2 joke behavior as an in-process SDK MCP tool."""
    payload = l2_random_joke()
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


def build_sdk_server(config: ClaudeSdkAgentConfig):
    """Build the in-process SDK MCP server for the current slice."""
    return create_sdk_mcp_server(
        name=config.server_name,
        version=config.server_version,
        tools=[random_joke_tool],
    )
