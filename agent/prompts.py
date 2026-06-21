from __future__ import annotations

"""Prompt construction helpers for the Weekend Wizard ReAct and reflection steps."""

from typing import Iterable, List

from agent.policies.guardrails import RequestAnalysis
from agent.tool_specs import TOOL_SPECS


def _tool_lines(tool_names: Iterable[str]) -> str:
    available = set(tool_names)
    ordered = [tool_name for tool_name in TOOL_SPECS if tool_name in available]
    lines = []
    for tool_name in ordered:
        spec = TOOL_SPECS[tool_name]
        lines.append(
            f"- {spec.name}\n"
            f"  purpose: {spec.purpose}\n"
            f"  input: {spec.input_contract}\n"
            f"  example: {spec.prompt_example}"
        )
    return "\n".join(lines)


def build_react_messages(
    planner_messages: List[dict[str, str]],
    tool_names: List[str],
    step_number: int,
    max_steps: int,
    request_analysis: RequestAnalysis | None = None,
) -> List[dict[str, str]]:
    """Build one bounded ReAct decision prompt for the LLM."""
    interpreted_request = ""
    if request_analysis is not None:
        interpreted_request = (
            "Interpreted request facts:\n"
            + "\n".join(request_analysis.summary_lines())
            + "\n"
        )

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
                "If an identical successful tool call already appears in observations, do not request it again; choose a different needed step or finish.\n"
                "Use the interpreted request facts below as the source of truth for what the user asked for and what dependencies already exist.\n"
                "Tool example:\n"
                '{"thought":"I need a joke first.","action":"tool","tool":"random_joke","args":{}}\n'
                "Early stop examples:\n"
                '- For "Tell me a joke.": call random_joke once, then finish.\n'
                '- For "Give me a trivia question.": call trivia once, then finish.\n'
                '- For "Give me weather and a joke.": get_weather, random_joke, then finish.\n'
                '- For "Plan a cozy Saturday in City A with weather and 3 mystery books.": city_to_coords if needed, get_weather, book_recs, then finish.\n'
                '- For "Get the weather for City A and City B.": fetch the needed weather results, then finish.\n'
                "Finish example:\n"
                '{"thought":"I have enough information.","action":"finish","final_answer":"Here is the comparison or final answer based on the gathered facts."}\n'
                f"{interpreted_request}"
                "Prior assistant decisions and tool results may be provided below.\n"
                "Use them as the record of what was already attempted and what happened before deciding the next step.\n"
                "Supported tools:\n"
                f"{_tool_lines(tool_names)}"
            ),
        },
    ]
    messages.extend(planner_messages)
    return messages


def build_reflection_messages(
    user_prompt: str,
    step_summary_lines: List[str],
    draft_answer: str,
) -> List[dict[str, str]]:
    """Build the one-shot reflection prompt."""
    observation_block = "\n".join(step_summary_lines) if step_summary_lines else "- none"

    return [
        {
            "role": "system",
            "content": (
                "You are composing the final grounded Weekend Wizard answer.\n"
                'Return ONLY valid JSON in the shape {"answer":"..."}.\n'
                "Write the final user-facing answer, not a raw intermediate dump.\n"
                "You may rewrite the structure into natural prose or cleaner bullets when helpful.\n"
                "Requirements:\n"
                "- preserve every concrete grounded fact already present in the draft\n"
                "- preserve failures, counts, locations, coordinates when needed for clarity, comparisons, and supported caveats\n"
                "- keep all requested result categories that were successfully fetched\n"
                "- keep the answer short, upbeat, complete, and grounded in observations\n"
                "- do not introduce new facts or suggest new tool calls\n"
                "- do not remove, soften, or reinterpret supported facts"
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
