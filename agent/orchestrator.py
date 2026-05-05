from __future__ import annotations

"""Planner/executor orchestration for one Weekend Wizard interaction."""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.grounding import (
    build_grounded_draft_from_payloads,
    parse_tool_payloads,
    parse_tool_payload_text,
)
from guardrails.execution import ExecutionStateSnapshot, normalize_tool_args
from guardrails.guardrails import parse_coords, requested_tools, validate_final_answer
from guardrails.plans import validate_plan_semantics
from agent.prompts import build_planner_messages, build_reflection_messages
from llm_client import llm_plan_json, llm_reflection_json
from logger.logging import get_log_extra, get_logger, staging_mode, telemetry_enabled
from mcp_runtime.client import ToolGateway, ToolInvocationError
from schemas.agent import (
    ExecutionPlan,
    InteractionResult,
    OrchestratorContext,
    ToolObservation,
)
from schemas.tools import GeoResult, ToolError


logger = get_logger("agent.orchestrator")


@dataclass
class ExecutionState:
    """Mutable execution state for one Weekend Wizard interaction.

    Attributes:
        plan: Validated execution plan produced by the planner.
        tool_observations: Structured tool outputs collected so far.
        resolved_coords: Coordinates resolved during execution, when available.
    """

    plan: ExecutionPlan
    tool_observations: List[ToolObservation]
    parsed_payloads: Dict[str, Any]
    resolved_coords: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class FinalizationAnalysis:
    """Parsed finalization view reused by draft building and failure detection."""

    payloads: Dict[str, Any]
    required_tool_failures: bool


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


def geo_payload_to_coords(payload: Any) -> Optional[Tuple[float, float]]:
    """Extract coordinates from a city lookup payload when possible."""
    if isinstance(payload, GeoResult):
        return payload.latitude, payload.longitude
    if isinstance(payload, ToolError):
        return None
    if isinstance(payload, dict):
        latitude = payload.get("latitude")
        longitude = payload.get("longitude")
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


def payload_indicates_error(payload: str) -> bool:
    """Return whether a serialized tool payload represents a tool failure."""
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return '"error"' in payload
    return isinstance(parsed, dict) and "error" in parsed


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
        if staging_mode():
            logger.info("Invoking tool %s with args=%s", tool_name, args)
        result = await tool_gateway.call_tool(tool_name, args)
        payload = render_tool_result(result)
        if staging_mode():
            logger.info("Tool %s completed", tool_name)
        return payload
    except ToolInvocationError as exc:
        logger.exception("Tool %s failed: %s", tool_name, exc)
        return _tool_error_payload(tool_name, str(exc))


def update_state_after_tool(state: ExecutionState, tool_name: str, payload: str) -> None:
    """Update execution state from a completed tool call.

    Args:
        state: Mutable execution state for the active interaction.
        tool_name: Name of the tool that just completed.
        payload: Serialized payload returned by the tool.
    """
    parsed_payload = parse_tool_payload_text(tool_name, payload)
    state.parsed_payloads[tool_name] = parsed_payload
    if tool_name == "city_to_coords":
        state.resolved_coords = geo_payload_to_coords(parsed_payload)
        if state.plan.location is not None and state.resolved_coords is not None:
            state.plan.location.latitude = state.resolved_coords[0]
            state.plan.location.longitude = state.resolved_coords[1]


def analyze_finalization(
    user_prompt: str,
    payloads: Dict[str, Any],
) -> FinalizationAnalysis:
    """Parse observations once for final grounded draft assembly and failure detection."""
    requested = requested_tools(user_prompt)
    required_tool_failures = False

    for tool_name, parsed_payload in payloads.items():
        if tool_name in requested:
            if isinstance(parsed_payload, ToolError):
                required_tool_failures = True
            elif isinstance(parsed_payload, str) and '"error"' in parsed_payload:
                required_tool_failures = True

    return FinalizationAnalysis(
        payloads=payloads,
        required_tool_failures=required_tool_failures,
    )


def _finalization_payloads(
    tool_observations: List[ToolObservation],
    parsed_payloads: Dict[str, Any],
) -> Dict[str, Any]:
    """Return canonical parsed payloads, reusing cache only when it matches observations."""
    canonical_payloads = parse_tool_payloads(
        {observation.tool_name: observation.payload for observation in tool_observations}
    )
    if not parsed_payloads:
        return canonical_payloads
    if set(parsed_payloads) != set(canonical_payloads):
        return canonical_payloads
    if any(parsed_payloads[tool_name] != canonical_payloads[tool_name] for tool_name in canonical_payloads):
        return canonical_payloads
    return parsed_payloads


def run_reflection(
    context: OrchestratorContext,
    user_prompt: str,
    payloads: Dict[str, Any],
    draft_answer: str,
) -> str:
    """Run one reflection pass and fall back to the grounded draft on failure.

    Args:
        context: Runtime interaction context carrying the active model name.
        user_prompt: Original user request.
        payloads: Parsed tool payloads collected during execution.
        draft_answer: Grounded draft answer to refine.

    Returns:
        The reflected answer when reflection succeeds, otherwise the original
        grounded draft answer.
    """
    messages = build_reflection_messages(user_prompt, payloads, draft_answer)
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
        return reflected.answer.strip()
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
    parsed_payloads: Dict[str, Any],
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
    parsed_payloads = _finalization_payloads(tool_observations, parsed_payloads)
    analysis = analyze_finalization(user_prompt, parsed_payloads)
    grounded = build_grounded_draft_from_payloads(user_prompt, "", analysis.payloads)
    reflected = run_reflection(context, user_prompt, analysis.payloads, grounded)
    if analysis.required_tool_failures:
        final_answer = grounded
        validation_failed = False
    else:
        validation = validate_final_answer(user_prompt, reflected, analysis.payloads)
        final_answer = reflected if validation.is_valid else grounded
        validation_failed = not validation.is_valid
    if not analysis.required_tool_failures and validation_failed:
        logger.warning(
            "Reflection answer missing observed content for tools=%s; falling back to grounded answer",
            ",".join(validation.missing_tools),
        )
    return build_interaction_result(
        context.history,
        answer=final_answer,
        tool_observations=tool_observations,
        used_fallback=used_fallback or analysis.required_tool_failures or validation_failed,
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
        plan = llm_plan_json(planner_messages, context.model_name)
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
            used_fallback=True,
        )

    logger.info(
        "Planner produced goal=%s with %d steps",
        plan.goal,
        len(plan.execution_steps),
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
        plan=plan,
        tool_observations=[],
        parsed_payloads={},
        resolved_coords=initial_coords,
    )

    tool_phase_started_at = time.perf_counter()
    for index, step in enumerate(plan.execution_steps, start=1):
        if staging_mode():
            logger.info("Executing planned step %d of %d: %s", index, len(plan.execution_steps), step.tool)
        tool_started_at = time.perf_counter()
        normalized_args, error = normalize_tool_args(
            step.tool,
            step.args,
            ExecutionStateSnapshot(
                plan=state.plan,
                resolved_coords=state.resolved_coords,
            ),
        )
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
                tool_outcome = "failure" if payload_indicates_error(payload) else "success"
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
        state.parsed_payloads,
        used_fallback=False,
    )
    interaction_outcome = "degraded" if result.used_fallback else "success"
    logger.info(
        "Interaction completed with %d observations and answer length %d",
        len(result.tool_observations),
        len(result.answer),
        extra=get_log_extra(
            event="interaction_completed",
            phase="orchestrator",
            outcome=interaction_outcome,
            duration_ms=round((time.perf_counter() - interaction_started_at) * 1000, 1),
            model_name=context.model_name,
        ),
    )
    return result
