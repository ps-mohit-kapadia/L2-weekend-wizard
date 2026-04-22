from __future__ import annotations

"""Planner/executor orchestration for one Weekend Wizard interaction."""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.grounding import (
    compose_grounded_answer_from_observations,
    parse_tool_payload_text,
)
from agent.guardrails.guardrails import parse_coords, requested_tools as infer_requested_tools
from agent.prompts import build_planner_messages, build_reflection_messages
from llm_client import llm_plan_json, llm_reflection_json
from logger.logging import get_log_extra, get_logger, telemetry_enabled
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
    """Mutable execution state for one Weekend Wizard interaction.

    Attributes:
        user_prompt: Original user prompt being processed.
        plan: Validated execution plan produced by the planner.
        tool_observations: Structured tool outputs collected so far.
        resolved_coords: Coordinates resolved during execution, when available.
    """

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


def validate_plan_semantics(plan: ExecutionPlan, available_tools: List[str], user_prompt: str) -> None:
    """Validate planner output against supported runtime constraints.

    Args:
        plan: Typed execution plan returned by the planner.
        available_tools: Tool names exposed by the MCP runtime.
        user_prompt: Original user prompt used to infer supported intent.

    Raises:
        ValueError: If the plan requests unsupported tools, omits requested tools,
            or violates runtime dependency rules.
    """
    available = set(available_tools)
    requested = set(plan.requested_tools)
    user_requested = infer_requested_tools(user_prompt)
    coords_in_prompt = parse_coords(user_prompt)
    if not requested:
        raise ValueError("Execution plan must request at least one tool.")
    if not requested.issubset(available):
        unknown = sorted(requested.difference(available))
        raise ValueError(f"Execution plan requested unsupported tools: {', '.join(unknown)}")
    if "city_to_coords" in requested:
        raise ValueError("Execution plan must not list city_to_coords as a requested tool.")
    extra_requested = requested.difference(user_requested)
    if extra_requested:
        raise ValueError(f"Execution plan added unrequested tools: {', '.join(sorted(extra_requested))}")

    seen_requested = set()
    saw_city_lookup = False
    weather_step_count = 0
    city_lookup_count = 0
    for step in plan.execution_steps:
        if step.tool not in available:
            raise ValueError(f"Execution step uses unsupported tool: {step.tool}")
        if step.tool != "city_to_coords":
            seen_requested.add(step.tool)
        if step.tool == "city_to_coords":
            saw_city_lookup = True
            city_lookup_count += 1
        if step.tool == "get_weather":
            weather_step_count += 1
            has_coords = (
                step.args.get("latitude") is not None
                and step.args.get("longitude") is not None
                or
                plan.location is not None
                and plan.location.latitude is not None
                and plan.location.longitude is not None
            )
            if not has_coords and not saw_city_lookup:
                raise ValueError("Weather execution requires coordinates or a prior city_to_coords step.")

    if city_lookup_count and "get_weather" not in requested:
        raise ValueError("city_to_coords is only valid as a dependency for weather requests.")
    if city_lookup_count and (
        coords_in_prompt is not None
        or (
            plan.location is not None
            and plan.location.latitude is not None
            and plan.location.longitude is not None
        )
    ):
        raise ValueError("Execution plan added city_to_coords even though coordinates were already provided.")
    if weather_step_count > 1 or city_lookup_count > 1:
        raise ValueError("Execution plan contains duplicate weather dependency steps.")

    missing_steps = requested.difference(seen_requested)
    if missing_steps:
        raise ValueError(f"Execution plan omitted requested tools: {', '.join(sorted(missing_steps))}")


def update_state_after_tool(state: ExecutionState, tool_name: str, payload: str) -> None:
    """Update execution state from a completed tool call.

    Args:
        state: Mutable execution state for the active interaction.
        tool_name: Name of the tool that just completed.
        payload: Serialized payload returned by the tool.
    """
    if tool_name == "city_to_coords":
        state.resolved_coords = geo_payload_to_coords(payload)
        if state.plan.location is not None and state.resolved_coords is not None:
            state.plan.location.latitude = state.resolved_coords[0]
            state.plan.location.longitude = state.resolved_coords[1]


def build_grounded_draft(user_prompt: str, tool_observations: List[ToolObservation]) -> str:
    """Build the grounded draft answer before reflection.

    Args:
        user_prompt: Original user request.
        tool_observations: Structured tool outputs collected during execution.

    Returns:
        A grounded draft answer derived from observed tool data.
    """
    return compose_grounded_answer_from_observations(user_prompt, "", tool_observations)


def run_reflection(
    context: OrchestratorContext,
    user_prompt: str,
    tool_observations: List[ToolObservation],
    draft_answer: str,
) -> str:
    """Run one reflection pass and fall back to the grounded draft on failure.

    Args:
        context: Runtime interaction context carrying the active model name.
        user_prompt: Original user request.
        tool_observations: Structured tool outputs collected during execution.
        draft_answer: Grounded draft answer to refine.

    Returns:
        The reflected answer when reflection succeeds, otherwise the original
        grounded draft answer.
    """
    messages = build_reflection_messages(user_prompt, tool_observations, draft_answer)
    started_at = time.perf_counter()
    try:
        reflected = llm_reflection_json(messages, context.model_name)
        if telemetry_enabled():
            logger.info(
                "Reflection completed",
                extra=get_log_extra(
                    event="reflection_completed",
                    phase="reflection",
                    outcome="success",
                    duration_ms=round((time.perf_counter() - started_at) * 1000, 1),
                    model_name=context.model_name,
                ),
            )
        return reflected["answer"].strip()
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started_at) * 1000, 1)
        logger.warning(
            "Reflection failed; returning grounded draft instead: %s",
            exc,
            extra=get_log_extra(
                event="reflection_failed",
                phase="reflection",
                outcome="fallback",
                duration_ms=duration_ms,
                model_name=context.model_name,
            ),
        )
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
    """Build, reflect, and persist the final answer after deterministic execution.

    Args:
        context: Runtime interaction context for the current request.
        user_prompt: Original user request.
        tool_observations: Structured tool outputs collected during execution.
        used_fallback: Whether the interaction completed through a fallback path.

    Returns:
        The final structured interaction result.
    """
    grounded = build_grounded_draft(user_prompt, tool_observations)
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
    """Run one planner/executor interaction from prompt to grounded result.

    Args:
        tool_gateway: MCP-backed tool execution gateway.
        context: Runtime interaction context for the current request.
        user_prompt: Original user request to process.

    Returns:
        The structured interaction result for the request.
    """
    interaction_started_at = time.perf_counter()
    logger.info(
        "Starting interaction for prompt length %d",
        len(user_prompt),
        extra=get_log_extra(event="interaction_started", phase="orchestrator", model_name=context.model_name),
    )
    context.history.append({"role": "user", "content": user_prompt})

    planner_messages = build_planner_messages(user_prompt, context.tool_names)
    planner_started_at = time.perf_counter()
    try:
        raw_plan = llm_plan_json(planner_messages, context.model_name)
        plan = validate_execution_plan(raw_plan)
        validate_plan_semantics(plan, context.tool_names, user_prompt)
    except Exception as exc:
        logger.exception(
            "Planning failed: %s",
            exc,
            extra=get_log_extra(
                event="planner_failed",
                phase="planner",
                outcome="failure",
                duration_ms=round((time.perf_counter() - planner_started_at) * 1000, 1),
                model_name=context.model_name,
            ),
        )
        return build_interaction_result(
            context.history,
            answer=build_planner_failure_answer(),
            tool_observations=[],
            used_fallback=False,
        )

    logger.info(
        "Planner produced goal=%s with %d steps and requested_tools=%s",
        plan.goal,
        len(plan.execution_steps),
        plan.requested_tools,
        extra=get_log_extra(
            event="planner_completed",
            phase="planner",
            outcome="success",
            duration_ms=round((time.perf_counter() - planner_started_at) * 1000, 1),
            model_name=context.model_name,
        ),
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

    tool_phase_started_at = time.perf_counter()
    for index, step in enumerate(plan.execution_steps, start=1):
        logger.info("Executing planned step %d of %d: %s", index, len(plan.execution_steps), step.tool)
        tool_started_at = time.perf_counter()
        normalized_args, error = normalize_tool_args(step.tool, step.args, state)
        if normalized_args is None:
            payload = _tool_error_payload(step.tool, error or "invalid args")
            logger.warning(
                "Tool step %s skipped due to invalid args",
                step.tool,
                extra=get_log_extra(
                    event="tool_failed",
                    phase="tool_execution",
                    outcome="invalid_args",
                    duration_ms=round((time.perf_counter() - tool_started_at) * 1000, 1),
                    tool_name=step.tool,
                ),
            )
        else:
            payload = await execute_tool_call(tool_gateway, step.tool, normalized_args)
            if telemetry_enabled():
                tool_outcome = "failure" if "\"error\"" in payload else "success"
                logger.info(
                    "Tool step completed: %s",
                    step.tool,
                    extra=get_log_extra(
                        event="tool_completed" if tool_outcome == "success" else "tool_failed",
                        phase="tool_execution",
                        outcome=tool_outcome,
                        duration_ms=round((time.perf_counter() - tool_started_at) * 1000, 1),
                        tool_name=step.tool,
                    ),
                )
        record_tool_observation(
            context.history,
            state.tool_observations,
            step.tool,
            normalized_args or step.args,
            payload,
        )
        update_state_after_tool(state, step.tool, payload)

    if telemetry_enabled():
        logger.info(
            "Tool execution phase completed",
            extra=get_log_extra(
                event="tool_phase_completed",
                phase="tool_execution",
                outcome="success",
                duration_ms=round((time.perf_counter() - tool_phase_started_at) * 1000, 1),
            ),
        )

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
        extra=get_log_extra(
            event="interaction_completed",
            phase="orchestrator",
            outcome="success",
            duration_ms=round((time.perf_counter() - interaction_started_at) * 1000, 1),
            model_name=context.model_name,
        ),
    )
    return result
