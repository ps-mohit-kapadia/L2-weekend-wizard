from __future__ import annotations

"""Prompt construction helpers for the Weekend Wizard planner and reflection steps."""

import json
from typing import Any, Dict, Iterable, List

from pydantic import BaseModel



_TOOL_SPECS = {
    "city_to_coords": 'city_to_coords args={"city":"New York"}',
    "get_weather": 'get_weather args={"latitude":40.7128,"longitude":-74.0060}',
    "book_recs": 'book_recs args={"topic":"mystery","limit":3}',
    "random_joke": "random_joke args={}",
    "random_dog": "random_dog args={}",
    "trivia": "trivia args={}",
}


def _tool_lines(tool_names: Iterable[str]) -> str:
    available = set(tool_names)
    ordered = [tool_name for tool_name in _TOOL_SPECS if tool_name in available]
    return "\n".join(f"- {_TOOL_SPECS[tool_name]}" for tool_name in ordered)


def build_planner_messages(
    user_prompt: str,
    tool_names: List[str],
) -> List[dict[str, str]]:
    """Build the one-shot planning prompt for the LLM."""
    return [
        {
            "role": "system",
            "content": (
                "You are Weekend Wizard's planning model.\n"
                "Your only job is to convert the user request into one structured execution plan.\n"
                "Do not answer the user.\n"
                "Do not explain your reasoning.\n"
                "Return ONLY valid JSON.\n"
                "Choose exactly one goal value from this list:\n"
                "- weekend_plan\n"
                "- weather_lookup\n"
                "- book_suggestions\n"
                "- joke\n"
                "- dog_photo\n"
                "- trivia\n"
                "JSON fields:\n"
                '- goal: one of the allowed values above\n'
                '- location: optional object with city, latitude, longitude\n'
                '- execution_steps: ordered executable tool steps\n'
                "Rules:\n"
                "Use only the supported tools listed below.\n"
                "Include only the minimum steps needed to satisfy the request.\n"
                "Do not include tools the user did not ask for.\n"
                "A weekend_plan goal does not authorize extra vibe tools by default.\n"
                "Do not add random_joke, random_dog, or trivia unless the user explicitly asked for them.\n"
                "Do not treat words like weekend, cozy, Saturday, Sunday, fun, or vibes as permission to add optional tools.\n"
                "If the user asks for books in a city, plan only book_recs unless weather or another tool was explicitly requested.\n"
                "Do not include trivia unless the user explicitly asked for trivia.\n"
                "If weather is requested and coordinates are already provided, use get_weather directly.\n"
                "If weather is requested and only a city is provided, add city_to_coords before get_weather.\n"
                "Do not add city_to_coords when valid coordinates are already provided.\n"
                "If books are requested, infer a concise topic and a reasonable limit.\n"
                "Single-tool requests should be the simplest case.\n"
                "If the user asks only for a joke, plan only random_joke.\n"
                "If the user asks only for a dog photo, plan only random_dog.\n"
                "If the user asks only for trivia, plan only trivia.\n"
                "If the user asks only for books, plan only book_recs.\n"
                "If the user asks only for weather and coordinates are already present, plan only get_weather.\n"
                'If the user asks "Plan a weekend in Las Vegas with 3 adventure books.", plan only book_recs.\n'
                "Never invent tools.\n"
                "Valid example:\n"
                '{'
                '"goal":"weekend_plan",'
                '"location":{"city":"New York","latitude":40.7128,"longitude":-74.0060},'
                '"execution_steps":['
                '{"tool":"get_weather","args":{"latitude":40.7128,"longitude":-74.0060}},'
                '{"tool":"book_recs","args":{"topic":"mystery","limit":3}},'
                '{"tool":"random_joke","args":{}},'
                '{"tool":"random_dog","args":{}}'
                ']'
                '}\n'
                "Single-tool example for trivia:\n"
                '{'
                '"goal":"trivia",'
                '"execution_steps":[{"tool":"trivia","args":{}}]'
                '}\n'
                "Books-only weekend-plan example:\n"
                '{'
                '"goal":"weekend_plan",'
                '"location":{"city":"Las Vegas"},'
                '"execution_steps":[{"tool":"book_recs","args":{"topic":"adventure","limit":3}}]'
                '}\n'
                "Supported tools:\n"
                f"{_tool_lines(tool_names)}"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def build_reflection_messages(
    user_prompt: str,
    payloads: Dict[str, Any],
    draft_answer: str,
) -> List[dict[str, str]]:
    """Build the one-shot reflection prompt."""
    observation_lines = [
        f"- {tool_name} payload={_format_reflection_payload(payload)}"
        for tool_name, payload in payloads.items()
    ]
    observation_block = "\n".join(observation_lines) if observation_lines else "- none"

    return [
        {
            "role": "system",
            "content": (
                "You are reviewing a draft Weekend Wizard answer.\n"
                'Return ONLY valid JSON in the shape {"answer":"..."}.\n'
                "Do one light reflection pass:\n"
                "- remove unsupported claims\n"
                "- make sure requested fetched items are reflected\n"
                "- keep the answer short, upbeat, and grounded in observations\n"
                "- do not introduce new facts or suggest new tool calls"
            ),
        },
        {
            "role": "user",
            "content": (
                f"User request:\n{user_prompt}\n\n"
                f"Tool observations:\n{observation_block}\n\n"
                f"Draft answer:\n{draft_answer}"
            ),
        },
    ]


def _format_reflection_payload(payload: Any) -> str:
    """Render one parsed payload into stable prompt text."""
    if isinstance(payload, BaseModel):
        return json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False)
