from __future__ import annotations

"""Prompt construction helpers for the Weekend Wizard ReAct and reflection steps."""

from typing import Iterable, List

from agent.grounding import render_compact_observation_summaries
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
    observation_summary: str | None = None,
) -> List[dict[str, str]]:
    """Build one bounded ReAct decision prompt for the LLM."""
    messages = [
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
                "Only act on entities, cities, locations, topics, or targets that the user explicitly requested or that a required tool directly returned.\n"
                "Never introduce a new city, topic, location, or target that the user did not ask for.\n"
                "Do not add extra enrichment, extra fun, or extra helpful information unless the user clearly asked for it.\n"
                "If the request is already satisfied by prior observations, choose finish immediately.\n"
                "Continuing to call tools after the request is satisfied is incorrect.\n"
                "Every tool call must be justified by the user's explicit request or a required dependency.\n"
                "Tools gather external facts only.\n"
                "Comparisons, summaries, recommendations, and final wording must happen in action=\"finish\" using final_answer.\n"
                "Only call one of the listed supported tools.\n"
                "Never invent tool names.\n"
                "For single-shot requests, one successful result is usually enough.\n"
                "Single-shot tools are random_joke, random_dog, and trivia.\n"
                "If the user asks for one joke, one dog photo, or one trivia question, call the tool once and then finish.\n"
                "Do not call the same single-shot tool again unless the user explicitly asked for multiple results or a retry.\n"
                "If weather is requested and coordinates are already available, prefer get_weather directly.\n"
                "If weather is requested and only a city is known, use city_to_coords before get_weather.\n"
                "If weather is requested for multiple locations, each requested location needs its own get_weather result before finish.\n"
                "city_to_coords resolves coordinates only.\n"
                "Resolving a city to coordinates is only a dependency, not fulfillment, when the user asked for weather.\n"
                "If the user asks for weather using coordinates, you must still call get_weather for each requested location after resolving coordinates.\n"
                "After city_to_coords succeeds for one requested location, call get_weather with that specific location's exact latitude and longitude.\n"
                "For multiple requested weather locations, collect one successful get_weather result per location, then finish.\n"
                "Do not finish while any requested or already-resolved location still lacks a weather observation.\n"
                "Do not repeat one location's weather if another requested location still needs weather.\n"
                "If an identical successful tool call already appears in observations, do not request it again; choose a different needed step or finish.\n"
                "Use weather only if the user asked for weather or a plan that depends on weather.\n"
                "Use books only if the user asked for books or a reading-themed plan.\n"
                "Use random_joke only if the user asked for a joke.\n"
                "Use random_dog only if the user asked for a dog photo or dog picture.\n"
                "Use trivia only if the user explicitly asked for trivia.\n"
                "Tool example:\n"
                '{"thought":"I need a joke first.","action":"tool","tool":"random_joke","args":{}}\n'
                "Early stop examples:\n"
                '- For "Tell me a joke.": call random_joke once, then finish.\n'
                '- For "Give me a trivia question.": call trivia once, then finish.\n'
                '- For "Give me weather and a joke.": get_weather, random_joke, then finish.\n'
                '- For "Plan a cozy Saturday in City A with weather and 3 mystery books.": city_to_coords if needed, get_weather, book_recs, then finish.\n'
                '- For "Get the weather for City A and City B using their coordinates.": city_to_coords for each location, then get_weather for each location, then finish.\n'
                "Finish example:\n"
                '{"thought":"I have enough information.","action":"finish","final_answer":"Here is the comparison or final answer based on the gathered facts."}\n'
                "Assistant observation context may be provided below after tools run.\n"
                "Use that context as the record of prior tool observations before deciding the next step.\n"
                "Supported tools:\n"
                f"{_tool_lines(tool_names)}"
            ),
        },
    ]
    if observation_summary:
        messages.append(
            {
                "role": "user",
                "content": f"Assistant observation context:\n{observation_summary}",
            }
        )
    messages.extend(history)
    return messages


def build_reflection_messages(
    user_prompt: str,
    tool_observations: List[ToolObservation],
    draft_answer: str,
) -> List[dict[str, str]]:
    """Build the one-shot reflection prompt."""
    observation_lines = render_compact_observation_summaries(user_prompt, tool_observations)
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
