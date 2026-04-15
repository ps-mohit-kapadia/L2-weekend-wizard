from __future__ import annotations

"""Prompt construction helpers for the Weekend Wizard agent."""

import json
from typing import Any, List


def build_system_prompt(tools: List[Any]) -> str:
    """Build the system prompt that teaches the model how to use MCP tools.

    Args:
        tools: Tool descriptors discovered from the MCP server.

    Returns:
        The system prompt supplied to the agent model.
    """
    tool_lines = []
    for tool in tools:
        description = (tool.description or "").strip()
        schema = json.dumps(tool.inputSchema, ensure_ascii=True)
        tool_lines.append(f"- {tool.name}: {description} | schema={schema}")

    return (
        "You are Weekend Wizard, a cheerful local assistant.\n"
        "Think step by step, but never reveal chain-of-thought.\n"
        "Decide whether to call one MCP tool or finish.\n"
        "You may call at most one tool per turn.\n"
        "When you need a tool, reply with ONLY valid JSON in this shape:\n"
        '{"action":"tool_name","args":{"param":"value"}}\n'
        "When you are ready to answer the user, reply with ONLY valid JSON in this shape:\n"
        '{"action":"final","answer":"short upbeat answer"}\n'
        "Use tools whenever the user asks for current weather, live recommendations, jokes, dog photos, or trivia.\n"
        "Reference fetched facts in the final answer.\n"
        "Available tools:\n"
        f"{chr(10).join(tool_lines)}"
    )
