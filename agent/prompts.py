from __future__ import annotations

"""Prompt construction helpers for the Weekend Wizard planner and reflection steps."""

from typing import Iterable, List

from schemas.agent import ToolObservation


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
                "Return ONLY valid JSON matching this schema:\n"
                '{'
                '"goal":"weekend_plan|weather_lookup|book_suggestions|joke|dog_photo|trivia",'
                '"location":{"city":"...","latitude":0.0,"longitude":0.0},'
                '"book_topic":"...",'
                '"requested_tools":["tool_name"],'
                '"execution_steps":[{"tool":"tool_name","args":{}}]'
                '}\n'
                "Use only the supported tools listed below.\n"
                "Include only the steps needed to satisfy the request.\n"
                "If weather is requested and coordinates are already provided, use get_weather directly.\n"
                "If weather is requested and only a city is provided, add city_to_coords before get_weather.\n"
                "If books are requested, infer a concise topic and a reasonable limit.\n"
                "Never invent tools.\n"
                "Supported tools:\n"
                f"{_tool_lines(tool_names)}"
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def build_reflection_messages(
    user_prompt: str,
    tool_observations: List[ToolObservation],
    draft_answer: str,
) -> List[dict[str, str]]:
    """Build the one-shot reflection prompt."""
    observation_lines = [
        f"- {observation.tool_name} args={observation.args} payload={observation.payload}"
        for observation in tool_observations
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
