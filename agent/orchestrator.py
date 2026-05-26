from __future__ import annotations

"""ReAct-style tool orchestration for one Weekend Wizard interaction."""

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.grounding import compose_grounded_answer_from_observations, parse_tool_payload_text
from agent.policies.guardrails import infer_book_limit, infer_book_topic, infer_city, parse_coords
from agent.prompts import build_react_messages, build_reflection_messages
from llm_client import llm_react_json, llm_reflection_json
from logger.logging import get_logger
from mcp_runtime.client import ToolGateway, ToolInvocationError
from schemas.agent import InteractionResult, OrchestratorContext, ReactDecision, ToolObservation, validate_react_decision
from schemas.tools import BookResults, DogResult, GeoResult, JokeResult, ToolError, TriviaResult, WeatherResult


logger = get_logger("agent.orchestrator")
MAX_REACT_STEPS = 6
SAFE_TOOL_INVOCATION_DETAIL = "tool execution failed"


@dataclass
class ExecutionState:
    """Mutable execution state for one Weekend Wizard interaction."""

    user_prompt: str
    tool_observations: List[ToolObservation]

    def candidate_weather_coords(self) -> List[Tuple[float, float]]:
        """Return deduplicated weather coordinate candidates from durable evidence."""
        candidates: "OrderedDict[Tuple[float, float], None]" = OrderedDict()

        prompt_coords = parse_coords(self.user_prompt)
        if prompt_coords is not None:
            candidates[(float(prompt_coords[0]), float(prompt_coords[1]))] = None

        for observation in self.tool_observations:
            if observation.tool_name != "city_to_coords":
                continue
            parsed = parse_tool_payload_text(observation.tool_name, observation.payload)
            if isinstance(parsed, GeoResult):
                candidates[(float(parsed.latitude), float(parsed.longitude))] = None

        return list(candidates.keys())

    def resolve_weather_coords(self) -> Tuple[Optional[Tuple[float, float]], Optional[str]]:
        """Resolve one unambiguous coordinate pair for an omitted weather request."""
        candidates = self.candidate_weather_coords()
        if not candidates:
            return None, "latitude and longitude are required"
        if len(candidates) > 1:
            return None, "latitude and longitude are required"
        return candidates[0], None


def render_tool_result(result: Any) -> str:
    """Serialize an MCP tool result into plain text for storage and grounding."""
    if getattr(result, "content", None):
        chunks: List[str] = []
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                chunks.append(text)
                continue
            if hasattr(item, "model_dump_json"):
                chunks.append(item.model_dump_json())
                continue
            chunks.append(str(item))
        if chunks:
            return "\n".join(chunks)

    if hasattr(result, "model_dump_json"):
        return result.model_dump_json()
    return str(result)

def build_interaction_result(
    history: List[Dict[str, str]],
    answer: str,
    tool_observations: List[ToolObservation],
    *,
    used_fallback: bool,
) -> InteractionResult:
    """Persist the final assistant answer and create the interaction result."""
    history.append({"role": "assistant", "content": answer})
    return InteractionResult(
        answer=answer,
        tool_observations=tool_observations,
        used_fallback=used_fallback,
    )


def _tool_error_payload(tool_name: str, details: str) -> str:
    return json.dumps({"error": f"{tool_name} failed", "details": details})


def record_tool_observation(
    tool_observations: List[ToolObservation],
    tool_name: str,
    args: Dict[str, Any],
    payload: str,
) -> None:
    """Record one structured tool observation for downstream summaries and grounding."""
    tool_observations.append(ToolObservation(tool_name=tool_name, args=args, payload=payload))


def has_successful_duplicate_observation(
    tool_observations: List[ToolObservation],
    tool_name: str,
    args: Dict[str, Any],
) -> bool:
    """Return whether an identical successful observation already exists."""
    for observation in tool_observations:
        if observation.tool_name != tool_name or observation.args != args:
            continue
        parsed = parse_tool_payload_text(observation.tool_name, observation.payload)
        if not isinstance(parsed, ToolError):
            return True
    return False


def build_observation_summary(tool_observations: List[ToolObservation]) -> str:
    """Build a compact structured summary for intermediate ReAct steps."""
    summary_lines: List[str] = []

    for observation in tool_observations:
        parsed = parse_tool_payload_text(observation.tool_name, observation.payload)
        tool_name = observation.tool_name

        if isinstance(parsed, ToolError):
            detail = parsed.details or parsed.error
            summary_lines.append(f"- {tool_name}: failed ({detail})")
            continue

        if tool_name == "city_to_coords" and isinstance(parsed, GeoResult):
            summary_lines.append(
                f"- city_to_coords: resolved {parsed.city} to {parsed.latitude}, {parsed.longitude}"
            )
            continue

        if tool_name == "get_weather" and isinstance(parsed, WeatherResult):
            detail = parsed.weather_summary or "weather fetched"
            temp = (
                f" at {parsed.temperature}{parsed.temperature_unit or ''}"
                if parsed.temperature is not None
                else ""
            )
            summary_lines.append(f"- get_weather: {detail}{temp}")
            continue

        if tool_name == "book_recs" and isinstance(parsed, BookResults):
            result_count = len(parsed.results)
            summary_lines.append(
                f"- book_recs: fetched {result_count} book recommendations for {parsed.topic}"
            )
            continue

        if tool_name == "random_joke" and isinstance(parsed, JokeResult):
            summary_lines.append("- random_joke: fetched one joke")
            continue

        if tool_name == "random_dog" and isinstance(parsed, DogResult):
            summary_lines.append("- random_dog: fetched one dog image")
            continue

        if tool_name == "trivia" and isinstance(parsed, TriviaResult):
            summary_lines.append("- trivia: fetched one trivia question")
            continue

        summary_lines.append(f"- {tool_name}: completed")

    return "\n".join(summary_lines)


async def execute_tool_call(
    tool_gateway: ToolGateway,
    tool_name: str,
    args: Dict[str, Any],
) -> str:
    """Invoke one MCP tool and serialize its response payload."""
    try:
        logger.info("Invoking tool %s with args=%s", tool_name, args)
        result = await tool_gateway.call_tool(tool_name, args)
        payload = render_tool_result(result)
        logger.info("Tool %s completed", tool_name)
        return payload
    except ToolInvocationError as exc:
        logger.exception("Tool %s failed: %s", tool_name, exc)
        return _tool_error_payload(tool_name, SAFE_TOOL_INVOCATION_DETAIL)


def normalize_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    state: ExecutionState,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Normalize and repair ReAct-produced tool args before execution."""
    args = dict(args or {})

    if tool_name == "city_to_coords":
        city = args.get("city") or infer_city(state.user_prompt)
        if not city:
            return None, "city is required"
        return {"city": str(city)}, None

    if tool_name == "get_weather":
        latitude = args.get("latitude")
        longitude = args.get("longitude")
        if latitude is None or longitude is None:
            coords, error = state.resolve_weather_coords()
            if coords is None:
                return None, error or "latitude and longitude are required"
            latitude, longitude = coords
        try:
            return {"latitude": float(latitude), "longitude": float(longitude)}, None
        except (TypeError, ValueError):
            return None, "latitude and longitude must be numeric"

    if tool_name == "book_recs":
        topic = args.get("topic") or args.get("param") or infer_book_topic(state.user_prompt)
        limit = args.get("limit") or infer_book_limit(state.user_prompt)
        if not topic:
            return None, "topic is required"
        try:
            safe_limit = max(1, min(int(limit), 10))
        except (TypeError, ValueError):
            safe_limit = 3
        return {"topic": str(topic), "limit": safe_limit}, None

    if tool_name in {"random_joke", "random_dog", "trivia"}:
        return {}, None

    return args, None


def validate_react_decision_semantics(
    decision: ReactDecision,
    available_tools: List[str],
) -> None:
    """Validate one ReAct decision against supported runtime constraints."""
    if decision.action == "tool":
        if not decision.tool:
            raise ValueError("Tool decisions must include a tool name.")
        if decision.tool not in available_tools:
            raise ValueError(f"Decision uses unsupported tool: {decision.tool}")
        return
    if not decision.final_answer or not decision.final_answer.strip():
        raise ValueError("Finish decisions must include a non-empty final answer.")

def build_grounded_draft(user_prompt: str, tool_observations: List[ToolObservation]) -> str:
    """Build the grounded draft answer before reflection."""
    return compose_grounded_answer_from_observations(user_prompt, "", tool_observations)


def run_reflection(
    context: OrchestratorContext,
    user_prompt: str,
    tool_observations: List[ToolObservation],
    draft_answer: str,
) -> Tuple[str, bool]:
    """Run one reflection pass and fall back to the grounded draft on failure."""
    messages = build_reflection_messages(user_prompt, tool_observations, draft_answer)
    try:
        reflected = llm_reflection_json(messages, context.model_name)
        return reflected["answer"].strip(), False
    except Exception as exc:
        logger.warning("Reflection failed; returning grounded draft instead: %s", exc)
        return draft_answer, True


def build_react_failure_answer() -> str:
    """Return a bounded failure message when the ReAct loop is not reliable."""
    return (
        "I couldn't complete a reliable weekend wizard turn for that yet. "
        "Try asking more directly for weather, book ideas, a joke, a dog photo, or trivia."
    )


def finalize_after_execution(
    context: OrchestratorContext,
    user_prompt: str,
    tool_observations: List[ToolObservation],
    draft_answer: str,
    *,
    used_fallback: bool = False,
) -> InteractionResult:
    """Build the grounded draft, reflect once, and persist the final answer.

    Reflection is the final natural-language writer. The grounded draft exists
    only to provide a safe evidence-based input and fallback when reflection
    fails.
    """
    grounded = (
        build_grounded_draft(user_prompt, tool_observations)
        if tool_observations
        else draft_answer
    )
    final_answer, reflection_used_fallback = run_reflection(context, user_prompt, tool_observations, grounded)
    return build_interaction_result(
        context.history,
        answer=final_answer,
        tool_observations=tool_observations,
        used_fallback=used_fallback or reflection_used_fallback,
    )


async def orchestrate_interaction(
    tool_gateway: ToolGateway,
    context: OrchestratorContext,
    user_prompt: str,
) -> InteractionResult:
    """Run one bounded ReAct interaction from prompt to grounded result."""
    logger.info("Starting interaction for prompt length %d", len(user_prompt))
    context.history.append({"role": "user", "content": user_prompt})

    state = ExecutionState(
        user_prompt=user_prompt,
        tool_observations=[],
    )
    draft_answer = ""

    for step_number in range(1, MAX_REACT_STEPS + 1):
        react_messages = build_react_messages(
            context.history,
            context.tool_names,
            step_number=step_number,
            max_steps=MAX_REACT_STEPS,
            observation_summary=build_observation_summary(state.tool_observations),
        )
        try:
            raw_decision = llm_react_json(
                react_messages,
                context.model_name,
                allowed_tools=context.tool_names,
            )
            decision = validate_react_decision(raw_decision)
            validate_react_decision_semantics(decision, context.tool_names)
        except Exception as exc:
            logger.exception("ReAct decision failed: %s", exc)
            if state.tool_observations:
                return finalize_after_execution(
                    context,
                    user_prompt,
                    state.tool_observations,
                    build_react_failure_answer(),
                    used_fallback=True,
                )
            return build_interaction_result(
                context.history,
                answer=build_react_failure_answer(),
                tool_observations=[],
                used_fallback=False,
            )

        if decision.action == "finish":
            draft_answer = decision.final_answer or ""
            break

        assert decision.tool is not None
        logger.info("Executing ReAct step %d of %d: %s", step_number, MAX_REACT_STEPS, decision.tool)
        normalized_args, error = normalize_tool_args(decision.tool, decision.args, state)
        if normalized_args is None:
            payload = _tool_error_payload(decision.tool, error or "invalid args")
        elif has_successful_duplicate_observation(state.tool_observations, decision.tool, normalized_args):
            logger.info(
                "Skipping duplicate successful tool call for %s with args=%s and finalizing",
                decision.tool,
                normalized_args,
            )
            return finalize_after_execution(
                context,
                user_prompt,
                state.tool_observations,
                draft_answer,
                used_fallback=False,
            )
        else:
            payload = await execute_tool_call(tool_gateway, decision.tool, normalized_args)
        record_tool_observation(
            state.tool_observations,
            decision.tool,
            normalized_args or decision.args,
            payload,
        )
    else:
        draft_answer = build_react_failure_answer()

    result = finalize_after_execution(
        context,
        user_prompt,
        state.tool_observations,
        draft_answer,
        used_fallback=False,
    )
    logger.info(
        "Interaction completed with %d observations and answer length %d",
        len(result.tool_observations),
        len(result.answer),
    )
    return result
