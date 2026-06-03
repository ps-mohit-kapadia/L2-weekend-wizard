from __future__ import annotations

"""ReAct-style tool orchestration for one Weekend Wizard interaction."""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.grounding import (
    compose_grounded_answer_from_observations,
    parse_tool_payload_text,
)
from agent.policies.guardrails import (
    infer_book_limit,
    infer_book_topic,
    infer_city,
)
from agent.prompts import build_react_messages, build_reflection_messages
from llm_client import llm_react_json, llm_reflection_json
from logger.logging import get_logger
from logger.tracing.request_trace import RequestTrace, truncate_repr
from mcp_runtime.client import ToolGateway, ToolInvocationError
from schemas.agent import (
    InteractionResult,
    OrchestratorContext,
    ReactDecision,
    ToolObservation,
    validate_react_decision,
)
from schemas.tools import ToolError


logger = get_logger("agent.orchestrator")
MAX_REACT_STEPS = 6
SAFE_TOOL_INVOCATION_DETAIL = "tool execution failed"


@dataclass
class ExecutionState:
    """Mutable execution state for one Weekend Wizard interaction."""

    user_prompt: str
    tool_observations: List[ToolObservation]


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
    tool_observations.append(
        ToolObservation(tool_name=tool_name, args=args, payload=payload)
    )


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


def _format_observation_args(args: Dict[str, Any]) -> str:
    """Render tool args compactly for planner-visible observation memory."""
    if not args:
        return "{}"
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def render_planner_observation_memory(
    tool_observations: List[ToolObservation],
) -> str:
    """Render planner-visible observation memory for ReAct planning.

    This is not completion logic and not a final answer draft. It serializes
    ToolObservation evidence into a compact faithful memory block so the
    planner can see prior tool executions without domain-specific narration.
    """
    summary_lines: List[str] = []

    for index, observation in enumerate(tool_observations, start=1):
        parsed = parse_tool_payload_text(observation.tool_name, observation.payload)
        tool_name = observation.tool_name
        args_text = _format_observation_args(observation.args)

        if isinstance(parsed, ToolError):
            detail = parsed.details or parsed.error
            summary_lines.append(
                f"[{index}] {tool_name}: failed ({detail}) args={args_text}"
            )
            continue

        summary_lines.append(f"[{index}] {tool_name}: completed args={args_text}")

    return "\n".join(summary_lines)


def render_assistant_observation_context(
    tool_observations: List[ToolObservation],
) -> str:
    """Backward-compatible alias for planner observation memory rendering."""
    return render_planner_observation_memory(tool_observations)


def build_observation_summary(tool_observations: List[ToolObservation]) -> str:
    """Backward-compatible alias for assistant observation context rendering."""
    return render_planner_observation_memory(tool_observations)


async def execute_tool_call(
    tool_gateway: ToolGateway,
    tool_name: str,
    args: Dict[str, Any],
    *,
    step_number: int,
    trace: RequestTrace | None = None,
) -> str:
    """Invoke one MCP tool and serialize its response payload."""
    try:
        if trace is not None:
            trace.add_event(
                "tool_execution_started",
                step_number=step_number,
                tool_name=tool_name,
                args=args,
            )
        logger.info("Invoking tool %s with args=%s", tool_name, args)
        started = time.perf_counter()
        result = await tool_gateway.call_tool(tool_name, args)
        payload = render_tool_result(result)
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Tool %s completed", tool_name)
        if trace is not None:
            trace.add_event(
                "tool_execution_completed",
                step_number=step_number,
                tool_name=tool_name,
                args=args,
                result=truncate_repr(payload),
                duration_ms=duration_ms,
            )
        return payload
    except ToolInvocationError as exc:
        logger.exception("Tool %s failed: %s", tool_name, exc)
        payload = _tool_error_payload(tool_name, SAFE_TOOL_INVOCATION_DETAIL)
        if trace is not None:
            duration_ms = int((time.perf_counter() - started) * 1000)
            trace.add_event(
                "tool_execution_completed",
                step_number=step_number,
                tool_name=tool_name,
                args=args,
                result=truncate_repr(payload),
                duration_ms=duration_ms,
            )
        return payload


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
            return None, "latitude and longitude are required"
        try:
            return {"latitude": float(latitude), "longitude": float(longitude)}, None
        except (TypeError, ValueError):
            return None, "latitude and longitude must be numeric"

    if tool_name == "book_recs":
        topic = (
            args.get("topic")
            or args.get("param")
            or infer_book_topic(state.user_prompt)
        )
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


def build_grounded_draft(
    user_prompt: str, tool_observations: List[ToolObservation]
) -> str:
    """Build the grounded draft answer before reflection."""
    return compose_grounded_answer_from_observations(user_prompt, "", tool_observations)


def run_reflection(
    context: OrchestratorContext,
    user_prompt: str,
    tool_observations: List[ToolObservation],
    draft_answer: str,
    *,
    trace: RequestTrace | None = None,
) -> Tuple[str, bool]:
    """Run one reflection pass and fall back to the grounded draft on failure."""
    messages = build_reflection_messages(user_prompt, tool_observations, draft_answer)
    try:
        reflected = llm_reflection_json(messages, context.model_name, trace=trace)
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
    trace: RequestTrace | None = None,
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
    final_answer, reflection_used_fallback = run_reflection(
        context, user_prompt, tool_observations, grounded, trace=trace
    )
    result = build_interaction_result(
        context.history,
        answer=final_answer,
        tool_observations=tool_observations,
        used_fallback=used_fallback or reflection_used_fallback,
    )
    if trace is not None:
        trace.add_event(
            "interaction_completed",
            observations_count=len(result.tool_observations),
            used_fallback=result.used_fallback,
            answer_length=len(result.answer),
        )
    return result


async def orchestrate_interaction(
    tool_gateway: ToolGateway,
    context: OrchestratorContext,
    user_prompt: str,
    *,
    trace: RequestTrace | None = None,
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
            observation_summary=render_planner_observation_memory(
                state.tool_observations
            ),
        )
        try:
            raw_decision = llm_react_json(
                react_messages,
                context.model_name,
                allowed_tools=context.tool_names,
                trace=trace,
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
                    trace=trace,
                )
            result = build_interaction_result(
                context.history,
                answer=build_react_failure_answer(),
                tool_observations=[],
                used_fallback=False,
            )
            if trace is not None:
                trace.add_event(
                    "interaction_completed",
                    observations_count=0,
                    used_fallback=False,
                    answer_length=len(result.answer),
                )
            return result

        if decision.action == "finish":
            draft_answer = decision.final_answer or ""
            break

        assert decision.tool is not None
        logger.info(
            "Executing ReAct step %d of %d: %s",
            step_number,
            MAX_REACT_STEPS,
            decision.tool,
        )
        normalized_args, error = normalize_tool_args(
            decision.tool, decision.args, state
        )
        if normalized_args is None:
            payload = _tool_error_payload(decision.tool, error or "invalid args")
        elif has_successful_duplicate_observation(
            state.tool_observations, decision.tool, normalized_args
        ):
            logger.info(
                "Skipping duplicate successful tool call for %s with args=%s and continuing",
                decision.tool,
                normalized_args,
            )
            if trace is not None:
                trace.add_event(
                    "duplicate_tool_call_skipped",
                    step_number=step_number,
                    tool_name=decision.tool,
                    args=normalized_args,
                    decision_summary=decision.thought,
                )
            continue
        else:
            payload = await execute_tool_call(
                tool_gateway,
                decision.tool,
                normalized_args,
                step_number=step_number,
                trace=trace,
            )
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
        trace=trace,
    )
    logger.info(
        "Interaction completed with %d observations and answer length %d",
        len(result.tool_observations),
        len(result.answer),
    )
    return result
