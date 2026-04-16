from __future__ import annotations

"""Planner/executor orchestration for one Weekend Wizard interaction."""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.grounding import (
    compose_grounded_answer_from_observations,
    parse_tool_payload_text,
)
from agent.prompts import build_planner_messages, build_reflection_messages
from llm_client import llm_plan_json, llm_reflection_json
from logger.logging import get_logger
from mcp_runtime.client import ToolGateway, ToolInvocationError
from schemas.agent import (
    ExecutionPlan,
    InteractionResult,
    OrchestratorContext,
    ToolObservation,
    validate_execution_plan,
)
from schemas.tools import GeoResult, ToolError


logger = get_logger("agent.orchestrator")


@dataclass
class ExecutionState:
    """Explicit execution state for one Weekend Wizard interaction."""

    user_prompt: str
    plan: ExecutionPlan
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
    used_step_limit_fallback: bool,
) -> InteractionResult:
    """Persist the final assistant answer and create the interaction result."""
    history.append({"role": "assistant", "content": answer})
    return InteractionResult(
        answer=answer,
        tool_observations=tool_observations,
        used_step_limit_fallback=used_step_limit_fallback,
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
    """Normalize and repair planned tool args before execution."""
    args = dict(args or {})

    if tool_name == "city_to_coords":
        city = args.get("city") or (state.plan.location.city if state.plan.location else None)
        if not city:
            return None, "city is required"
        return {"city": str(city)}, None

    if tool_name == "get_weather":
        latitude = args.get("latitude")
        longitude = args.get("longitude")
        if latitude is None or longitude is None:
            coords = state.resolved_coords
            if coords is not None:
                latitude, longitude = coords
            elif state.plan.location and state.plan.location.latitude is not None and state.plan.location.longitude is not None:
                latitude, longitude = state.plan.location.latitude, state.plan.location.longitude
        if latitude is None or longitude is None:
            return None, "latitude and longitude are required"
        try:
            return {"latitude": float(latitude), "longitude": float(longitude)}, None
        except (TypeError, ValueError):
            return None, "latitude and longitude must be numeric"

    if tool_name == "book_recs":
        topic = args.get("topic") or args.get("param") or state.plan.book_topic
        limit = args.get("limit") or 3
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


def validate_plan_semantics(plan: ExecutionPlan, available_tools: List[str]) -> None:
    """Validate planner output against supported runtime constraints."""
    available = set(available_tools)
    requested = set(plan.requested_tools)
    if not requested:
        raise ValueError("Execution plan must request at least one tool.")
    if not requested.issubset(available):
        unknown = sorted(requested.difference(available))
        raise ValueError(f"Execution plan requested unsupported tools: {', '.join(unknown)}")

    seen_requested = set()
    saw_city_lookup = False
    for step in plan.execution_steps:
        if step.tool not in available:
            raise ValueError(f"Execution step uses unsupported tool: {step.tool}")
        seen_requested.add(step.tool)
        if step.tool == "city_to_coords":
            saw_city_lookup = True
        if step.tool == "get_weather":
            has_coords = (
                plan.location is not None
                and plan.location.latitude is not None
                and plan.location.longitude is not None
            )
            if not has_coords and not saw_city_lookup:
                raise ValueError("Weather execution requires coordinates or a prior city_to_coords step.")

    missing_steps = requested.difference(seen_requested)
    if missing_steps:
        raise ValueError(f"Execution plan omitted requested tools: {', '.join(sorted(missing_steps))}")


def update_state_after_tool(state: ExecutionState, tool_name: str, payload: str) -> None:
    """Update execution state from a tool result."""
    if tool_name == "city_to_coords":
        state.resolved_coords = geo_payload_to_coords(payload)
        if state.plan.location is not None and state.resolved_coords is not None:
            state.plan.location.latitude = state.resolved_coords[0]
            state.plan.location.longitude = state.resolved_coords[1]


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


def build_planner_failure_answer() -> str:
    """Return a bounded failure message when planning is not reliable."""
    return (
        "I couldn't build a reliable weekend plan for that yet. "
        "Try asking more directly for weather, book ideas, a joke, a dog photo, or trivia."
    )


def finalize_after_execution(
    context: OrchestratorContext,
    user_prompt: str,
    tool_observations: List[ToolObservation],
    *,
    used_fallback: bool = False,
) -> InteractionResult:
    """Build, reflect, and persist the final answer after deterministic execution."""
    grounded = build_grounded_draft(user_prompt, tool_observations)
    final_answer = run_reflection(context, user_prompt, tool_observations, grounded)
    final_answer = compose_grounded_answer_from_observations(user_prompt, final_answer, tool_observations)
    return build_interaction_result(
        context.history,
        answer=final_answer,
        tool_observations=tool_observations,
        used_step_limit_fallback=used_fallback,
    )


async def orchestrate_interaction(
    tool_gateway: ToolGateway,
    context: OrchestratorContext,
    user_prompt: str,
) -> InteractionResult:
    """Run one planner/executor interaction from prompt to grounded result."""
    logger.info("Starting interaction for prompt length %d", len(user_prompt))
    context.history.append({"role": "user", "content": user_prompt})

    planner_messages = build_planner_messages(user_prompt, context.tool_names)
    try:
        raw_plan = llm_plan_json(planner_messages, context.model_name)
        plan = validate_execution_plan(raw_plan)
        validate_plan_semantics(plan, context.tool_names)
    except Exception as exc:
        logger.exception("Planning failed: %s", exc)
        return build_interaction_result(
            context.history,
            answer=build_planner_failure_answer(),
            tool_observations=[],
            used_step_limit_fallback=False,
        )

    logger.info(
        "Planner produced goal=%s with %d steps and requested_tools=%s",
        plan.goal,
        len(plan.execution_steps),
        plan.requested_tools,
    )

    initial_coords = None
    if plan.location and plan.location.latitude is not None and plan.location.longitude is not None:
        initial_coords = (plan.location.latitude, plan.location.longitude)
    state = ExecutionState(
        user_prompt=user_prompt,
        plan=plan,
        tool_observations=[],
        resolved_coords=initial_coords,
    )

    for index, step in enumerate(plan.execution_steps, start=1):
        logger.info("Executing planned step %d of %d: %s", index, len(plan.execution_steps), step.tool)
        normalized_args, error = normalize_tool_args(step.tool, step.args, state)
        if normalized_args is None:
            payload = _tool_error_payload(step.tool, error or "invalid args")
        else:
            payload = await execute_tool_call(tool_gateway, step.tool, normalized_args)
        record_tool_observation(
            context.history,
            state.tool_observations,
            step.tool,
            normalized_args or step.args,
            payload,
        )
        update_state_after_tool(state, step.tool, payload)

    result = finalize_after_execution(
        context,
        user_prompt,
        state.tool_observations,
        used_fallback=False,
    )
    logger.info(
        "Interaction completed with %d observations and answer length %d",
        len(result.tool_observations),
        len(result.answer),
    )
    return result
