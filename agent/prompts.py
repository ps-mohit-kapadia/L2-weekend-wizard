from __future__ import annotations

"""Prompt construction helpers for the Weekend Wizard ReAct and reflection steps."""

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


def build_react_messages(
    history: List[dict[str, str]],
    tool_names: List[str],
    step_number: int,
    max_steps: int,
) -> List[dict[str, str]]:
    """Build one bounded ReAct decision prompt for the LLM."""
    return [
        {
            "role": "system",
            "content": (
                "You are Weekend Wizard, a small local ReAct-style weekend helper.\n"
                "You think one step at a time.\n"
                "Return ONLY valid JSON.\n"
                "JSON shape:\n"
                '- thought: short reasoning for the next step\n'
                '- action: "tool" or "finish"\n'
                '- tool: required only when action is "tool"\n'
                '- args: object, required only when action is "tool"\n'
                '- final_answer: required only when action is "finish"\n'
                f"You are on step {step_number} of at most {max_steps}.\n"
                "Use the minimum number of tool calls needed.\n"
                "Only call tools that are necessary to satisfy the user's explicit request.\n"
                "Do not add extra enrichment, extra fun, or extra helpful information unless the user clearly asked for it.\n"
                "If the request is already satisfied by prior observations, choose finish immediately.\n"
                "Continuing to call tools after the request is satisfied is incorrect.\n"
                "Every tool call must be justified by the user's explicit request or a required dependency.\n"
                "If weather is requested and coordinates are already available, prefer get_weather directly.\n"
                "If weather is requested and only a city is known, use city_to_coords before get_weather.\n"
                "Do not repeat a tool call if a prior observation already satisfies that need.\n"
                "Use weather only if the user asked for weather or a plan that depends on weather.\n"
                "Use books only if the user asked for books or a reading-themed plan.\n"
                "Use random_joke only if the user asked for a joke.\n"
                "Use random_dog only if the user asked for a dog photo or dog picture.\n"
                "Use trivia only if the user explicitly asked for trivia.\n"
                "Never invent tools.\n"
                "Tool example:\n"
                '{"thought":"I need a joke first.","action":"tool","tool":"random_joke","args":{}}\n'
                "Early stop examples:\n"
                '- For "Tell me a joke.": call random_joke, then finish.\n'
                '- For "Give me weather and a joke.": get_weather, random_joke, then finish.\n'
                '- For "Plan a cozy Saturday in New York with weather and 3 mystery books.": city_to_coords if needed, get_weather, book_recs, then finish.\n'
                "Finish example:\n"
                '{"thought":"I have enough information.","action":"finish","final_answer":"Here is a cozy weekend plan for you..."}\n'
                "Any assistant message in the form [tool:name] payload is a previous tool observation.\n"
                "Use those observations before deciding the next step.\n"
                "Supported tools:\n"
                f"{_tool_lines(tool_names)}"
            ),
        },
        *history,
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
