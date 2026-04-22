from __future__ import annotations

"""ReAct-style tool orchestration for one Weekend Wizard interaction."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.grounding import compose_grounded_answer_from_observations, parse_tool_payload_text
from agent.policies.guardrails import infer_book_limit, infer_book_topic, infer_city, parse_coords
from agent.prompts import build_react_messages, build_reflection_messages
from llm_client import llm_react_json, llm_reflection_json
from logger.logging import get_logger
from mcp_runtime.client import ToolGateway, ToolInvocationError
from schemas.agent import InteractionResult, OrchestratorContext, ReactDecision, ToolObservation, validate_react_decision
from schemas.tools import GeoResult, ToolError


logger = get_logger("agent.orchestrator")
MAX_REACT_STEPS = 6


@dataclass
class ExecutionState:
    """Mutable execution state for one Weekend Wizard interaction."""

    user_prompt: str
    tool_observations: List[ToolObservation]
    resolved_coords: Optional[Tuple[float, float]]


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


def geo_payload_to_coords(payload: str) -> Optional[Tuple[float, float]]:
    """Extract coordinates from a serialized city lookup payload when possible."""
    parsed = parse_tool_payload_text("city_to_coords", payload)
    if isinstance(parsed, GeoResult):
        return parsed.latitude, parsed.longitude
    if isinstance(parsed, ToolError):
        return None
    if isinstance(parsed, dict):
        latitude = parsed.get("latitude")
        longitude = parsed.get("longitude")
        if isinstance(latitude, (int, float)) and isinstance(longitude, (int, float)):
            return float(latitude), float(longitude)
    return None


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
    history: List[Dict[str, str]],
    tool_observations: List[ToolObservation],
    tool_name: str,
    args: Dict[str, Any],
    payload: str,
) -> None:
    """Record a tool observation in both free-form and structured interaction state."""
    tool_observations.append(ToolObservation(tool_name=tool_name, args=args, payload=payload))
    history.append({"role": "assistant", "content": f"[tool:{tool_name}] {payload}"})


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
        return _tool_error_payload(tool_name, str(exc))


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
            coords = state.resolved_coords or parse_coords(state.user_prompt)
            if coords is not None:
                latitude, longitude = coords
        if latitude is None or longitude is None:
            return None, "latitude and longitude are required"
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


def update_state_after_tool(state: ExecutionState, tool_name: str, payload: str) -> None:
    """Update execution state from a completed tool call."""
    if tool_name == "city_to_coords":
        state.resolved_coords = geo_payload_to_coords(payload)


def build_grounded_draft(user_prompt: str, tool_observations: List[ToolObservation]) -> str:
    """Build the grounded draft answer before reflection."""
    return compose_grounded_answer_from_observations(user_prompt, "", tool_observations)


def run_reflection(
    context: OrchestratorContext,
    user_prompt: str,
    tool_observations: List[ToolObservation],
    draft_answer: str,
) -> str:
    """Run one reflection pass and fall back to the grounded draft on failure."""
    messages = build_reflection_messages(user_prompt, tool_observations, draft_answer)
    try:
        reflected = llm_reflection_json(messages, context.model_name)
        return reflected["answer"].strip()
    except Exception as exc:
        logger.warning("Reflection failed; returning grounded draft instead: %s", exc)
        return draft_answer


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
    """Build, reflect, and persist the final answer after execution."""
    grounded = draft_answer or build_grounded_draft(user_prompt, tool_observations)
    final_answer = run_reflection(context, user_prompt, tool_observations, grounded)
    final_answer = compose_grounded_answer_from_observations(user_prompt, final_answer, tool_observations)
    return build_interaction_result(
        context.history,
        answer=final_answer,
        tool_observations=tool_observations,
        used_fallback=used_fallback,
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
        resolved_coords=parse_coords(user_prompt),
    )
    draft_answer = ""

    for step_number in range(1, MAX_REACT_STEPS + 1):
        react_messages = build_react_messages(
            context.history,
            context.tool_names,
            step_number=step_number,
            max_steps=MAX_REACT_STEPS,
        )
        try:
            raw_decision = llm_react_json(react_messages, context.model_name)
            decision = validate_react_decision(raw_decision)
            validate_react_decision_semantics(decision, context.tool_names)
        except Exception as exc:
            logger.exception("ReAct decision failed: %s", exc)
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
        else:
            payload = await execute_tool_call(tool_gateway, decision.tool, normalized_args)
        record_tool_observation(
            context.history,
            state.tool_observations,
            decision.tool,
            normalized_args or decision.args,
            payload,
        )
        update_state_after_tool(state, decision.tool, payload)
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
