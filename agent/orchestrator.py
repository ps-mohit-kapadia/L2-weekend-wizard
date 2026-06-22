from __future__ import annotations

"""ReAct-style tool orchestration for one Weekend Wizard interaction."""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from agent.grounding import (
    compose_grounded_answer_from_steps,
    render_reflection_observations,
    render_tool_feedback,
)
from agent.tool_specs import get_tool_spec
from agent.policies.guardrails import (
    RequestAnalysis,
    analyze_request,
    infer_book_limit,
    infer_book_topic,
    infer_city,
)
from agent.prompts import REACT_PROMPT_CONTRACT, REFLECTION_PROMPT_CONTRACT, build_react_messages, build_reflection_messages
from llm_client import llm_react_json, llm_reflection_json
from logger.logging import get_logger
from logger.tracing.request_trace import RequestTrace, trace_llm_decision, traced_span, truncate_repr
from mcp_runtime.client import ToolGateway, ToolInvocationError, extract_tool_payload
from schemas.agent import (
    InteractionResult,
    OrchestratorContext,
    ReactDecision,
    ReflectionResult,
    ToolObservation,
    validate_react_decision,
)
from schemas.tools import (
    BookArgs,
    CityArgs,
    EmptyArgs,
    ToolError,
    ToolArgs,
    WeatherArgs,
    dump_tool_args,
    parse_tool_payload,
)


logger = get_logger("agent.orchestrator")
MAX_REACT_STEPS = 6
SAFE_TOOL_INVOCATION_DETAIL = "tool execution failed"


@dataclass
class ExecutionStep:
    """One semantic execution step within a single interaction."""

    kind: str
    thought: str = ""
    tool_name: str = ""
    raw_args: Dict[str, Any] | None = None
    normalized_args: ToolArgs | None = None
    parsed_payload: Any = None
    outcome: str = ""
    feedback_message: str = ""
    final_answer: str = ""


@dataclass
class FulfillmentState:
    """Fulfillment status for one requested result category."""

    requested: bool = True
    fulfilled: bool = False
    degraded: bool = False


@dataclass
class ExecutionState:
    """Single semantic source of truth for one Weekend Wizard interaction."""

    user_prompt: str
    steps: List[ExecutionStep]
    request_analysis: RequestAnalysis | None = None
    fulfillment: Dict[str, FulfillmentState] | None = None
    final_answer: str = ""
    used_fallback: bool = False


def _serialize_tool_payload(payload: Any) -> str:
    """Serialize one extracted tool payload for traces and output compatibility."""
    if hasattr(payload, "model_dump"):
        return json.dumps(payload.model_dump())
    if hasattr(payload, "model_dump_json"):
        dumped = payload.model_dump_json()
        try:
            return json.dumps(json.loads(dumped))
        except json.JSONDecodeError:
            return dumped
    if isinstance(payload, (dict, list)):
        return json.dumps(payload)
    return str(payload)


def _initialize_fulfillment(
    request_analysis: RequestAnalysis | None,
) -> Dict[str, FulfillmentState]:
    """Initialize fulfillment tracking from interpreted requested categories."""
    if request_analysis is None:
        return {}
    return {
        tool_name: FulfillmentState()
        for tool_name in sorted(request_analysis.requested_tools)
    }


def _update_fulfillment(state: ExecutionState, tool_name: str, payload: Any) -> None:
    """Update fulfillment state after one parsed tool result."""
    spec = get_tool_spec(tool_name)
    if not spec.fulfills_requested_work:
        return
    status = (state.fulfillment or {}).get(tool_name)
    if status is None:
        return
    if isinstance(payload, ToolError):
        if not status.fulfilled:
            status.degraded = True
        return
    status.fulfilled = True
    status.degraded = False


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


def _derive_tool_observations(state: ExecutionState) -> List[ToolObservation]:
    """Project tool observations from semantic execution steps."""
    observations: List[ToolObservation] = []
    for step in state.steps:
        if step.kind not in {"tool_call", "tool_invalid"} or not step.tool_name:
            continue
        payload = _serialize_tool_payload(step.parsed_payload)
        observations.append(
            ToolObservation(
                tool_name=step.tool_name,
                args=dump_tool_args(step.normalized_args or step.raw_args),
                payload=payload,
            )
        )
    return observations


def _render_planner_messages(state: ExecutionState) -> List[Dict[str, str]]:
    """Render planner-visible transcript from semantic execution state."""
    messages: List[Dict[str, str]] = [{"role": "user", "content": state.user_prompt}]
    for step in state.steps:
        if step.kind == "finish_blocked":
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Thought: {step.thought}\n"
                        "Action: finish\n"
                        f"Final Answer: {step.final_answer}"
                    ),
                }
            )
            if step.feedback_message:
                messages.append({"role": "tool", "content": step.feedback_message})
            continue
        if step.kind not in {"tool_call", "tool_invalid", "tool_skip"}:
            continue
        planner_args = dump_tool_args(step.normalized_args or step.raw_args)
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"Thought: {step.thought}\n"
                    "Action: tool\n"
                    f"Tool: {step.tool_name}\n"
                    f"Args: {_format_observation_args(planner_args)}"
                ),
            }
        )
        if step.feedback_message:
            messages.append({"role": "tool", "content": step.feedback_message})
    return messages


def has_successful_duplicate_observation(
    steps: List[ExecutionStep],
    tool_name: str,
    args: ToolArgs,
) -> bool:
    """Return whether an identical successful observation already exists."""
    expected_args = dump_tool_args(args)
    for step in steps:
        if step.kind != "tool_call":
            continue
        observed_args = dump_tool_args(step.normalized_args or step.raw_args)
        if step.tool_name != tool_name or observed_args != expected_args:
            continue
        if not isinstance(step.parsed_payload, ToolError):
            return True
    return False


def _format_observation_args(args: ToolArgs | Dict[str, Any] | None) -> str:
    """Render tool args compactly for planner-visible assistant/tool messages."""
    dumped = dump_tool_args(args)
    if not dumped:
        return "{}"
    return json.dumps(dumped, sort_keys=True, separators=(",", ":"))


def _planner_message_signature(tool_name: str, args: ToolArgs | Dict[str, Any] | None) -> str:
    return f"{tool_name}:{_format_observation_args(args)}"


def _planner_tool_feedback_detail(
    tool_name: str,
    args: ToolArgs | Dict[str, Any],
    parsed: Any,
) -> str:
    """Render planner-local tool feedback from typed tool output."""
    return render_tool_feedback(tool_name, args, parsed)


@traced_span(
    "tool.call",
    lambda args: {
        "tool_name": args["tool_name"],
        "step_number": args["step_number"],
    },
)
async def execute_tool_call(
    tool_gateway: ToolGateway,
    tool_name: str,
    args: Dict[str, Any],
    *,
    step_number: int,
    trace: RequestTrace | None = None,
) -> Any:
    """Invoke one MCP tool and return the raw extracted payload plus trace text."""
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
        raw_payload = extract_tool_payload(result)
        duration_ms = int((time.perf_counter() - started) * 1000)
        logger.info("Tool %s completed", tool_name)
        if trace is not None:
            trace.add_event(
                "tool_execution_completed",
                step_number=step_number,
                tool_name=tool_name,
                args=args,
                result=truncate_repr(_serialize_tool_payload(raw_payload)),
                duration_ms=duration_ms,
            )
        return raw_payload
    except ToolInvocationError as exc:
        logger.exception("Tool %s failed: %s", tool_name, exc)
        raw_payload = {"error": f"{tool_name} failed", "details": SAFE_TOOL_INVOCATION_DETAIL}
        if trace is not None:
            duration_ms = int((time.perf_counter() - started) * 1000)
            trace.add_event(
                "tool_execution_completed",
                step_number=step_number,
                tool_name=tool_name,
                args=args,
                result=truncate_repr(_serialize_tool_payload(raw_payload)),
                duration_ms=duration_ms,
            )
        return raw_payload


def normalize_tool_args(
    tool_name: str,
    args: Dict[str, Any],
    state: ExecutionState,
) -> Tuple[Optional[ToolArgs], Optional[str]]:
    """Normalize and repair ReAct-produced tool args before execution."""
    args = dict(args or {})
    spec = get_tool_spec(tool_name)

    if spec.arg_policy == "city":
        city = (
            args.get("city")
            or (state.request_analysis.city if state.request_analysis is not None else None)
            or infer_city(state.user_prompt)
        )
        if not city:
            return None, "city is required"
        return CityArgs(city=str(city)), None

    if spec.arg_policy == "weather_coords":
        latitude = args.get("latitude")
        longitude = args.get("longitude")
        if latitude is None or longitude is None:
            return None, "latitude and longitude are required"
        try:
            return WeatherArgs(latitude=float(latitude), longitude=float(longitude)), None
        except (TypeError, ValueError):
            return None, "latitude and longitude must be numeric"

    if spec.arg_policy == "book_recs":
        topic = (
            args.get("topic")
            or args.get("param")
            or (state.request_analysis.book_topic if state.request_analysis is not None else None)
            or infer_book_topic(state.user_prompt)
        )
        limit = (
            (state.request_analysis.book_limit if state.request_analysis is not None else None)
            or args.get("limit")
            or infer_book_limit(state.user_prompt)
        )
        if not topic:
            return None, "topic is required"
        try:
            safe_limit = max(1, min(int(limit), 10))
        except (TypeError, ValueError):
            safe_limit = 3
        return BookArgs(topic=str(topic), limit=safe_limit), None

    if spec.arg_policy == "empty":
        return EmptyArgs(), None

    return EmptyArgs(), None


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
    user_prompt: str,
    state: ExecutionState,
) -> str:
    """Build the grounded draft answer before reflection."""
    rendered_steps = [
        step
        for step in state.steps
        if not step.tool_name or get_tool_spec(step.tool_name).fulfills_requested_work
    ]
    return compose_grounded_answer_from_steps(user_prompt, "", rendered_steps)


def _category_label(tool_name: str) -> str:
    return get_tool_spec(tool_name).category_label


def _pending_requested_categories(state: ExecutionState) -> List[str]:
    pending: List[str] = []
    for tool_name, status in (state.fulfillment or {}).items():
        if not status.fulfilled and not status.degraded:
            pending.append(tool_name)
    return pending


def run_reflection(
    context: OrchestratorContext,
    user_prompt: str,
    state: ExecutionState,
    draft_answer: str,
    *,
    trace: RequestTrace | None = None,
) -> Tuple[ReflectionResult | None, bool]:
    """Run one quality pass over the grounded draft and return review metadata."""
    step_summary_lines = render_reflection_observations(user_prompt, state.steps)
    messages = build_reflection_messages(user_prompt, step_summary_lines, draft_answer)
    try:
        if trace is None:
            reflected = llm_reflection_json(
                messages,
                context.model_name,
                trace=trace,
                prompt_contract=REFLECTION_PROMPT_CONTRACT,
            )
        else:
            with trace.span("reflection.call", observations_count=len(step_summary_lines)):
                reflected = llm_reflection_json(
                    messages,
                    context.model_name,
                    trace=trace,
                    prompt_contract=REFLECTION_PROMPT_CONTRACT,
                )
        reflection = (
            reflected
            if isinstance(reflected, ReflectionResult)
            else ReflectionResult.model_validate(reflected)
        )
        if trace is not None:
            trace.add_event(
                "reflection.reviewed",
                verdict=reflection.verdict,
                issues_count=len(reflection.issues),
                intro_preview=truncate_repr(reflection.intro, 120),
                outro_preview=truncate_repr(reflection.outro, 120),
            )
        return reflection, False
    except Exception as exc:
        logger.warning(
            "Reflection failed; returning grounded draft instead: %s | event=reflection.failed reason=reflection_error fallback=grounded_draft",
            exc,
        )
        trace_llm_decision(
            trace,
            phase="reflection",
            action="accept",
            accepted=False,
            reason="reflection_error",
        )
        return None, True


def _compose_final_answer(grounded: str, reflection: ReflectionResult | None) -> str:
    """Compose final output while keeping grounded facts deterministic."""
    if reflection is None or reflection.verdict != "pass":
        return grounded
    parts = [part.strip() for part in (reflection.intro, grounded, reflection.outro) if part and part.strip()]
    return "\n\n".join(parts) if parts else grounded


def build_react_failure_answer() -> str:
    """Return a bounded failure message when the ReAct loop is not reliable."""
    return (
        "I couldn't complete a reliable weekend wizard turn for that yet. "
        "Try asking more directly for weather, book ideas, a joke, a dog photo, or trivia."
    )


def finalize_after_execution(
    context: OrchestratorContext,
    user_prompt: str,
    state: ExecutionState,
    draft_answer: str,
    *,
    used_fallback: bool = False,
    trace: RequestTrace | None = None,
) -> InteractionResult:
    """Build grounded output first, then run one quality-only reflection pass.

    The grounded draft is the semantic source for tool-backed turns. Reflection
    is allowed to improve presentation quality, but grounded output remains the
    authority baseline when reflection fails or drifts away from grounded facts.
    """
    if trace is None:
        grounded = (
            build_grounded_draft(user_prompt, state)
            if _derive_tool_observations(state)
            else draft_answer
        )
    else:
        with trace.span("grounding.build"):
            grounded = (
                build_grounded_draft(user_prompt, state)
                if _derive_tool_observations(state)
                else draft_answer
            )
    reflection_result = run_reflection(
        context, user_prompt, state, grounded, trace=trace
    )
    reflection, reflection_used_fallback = reflection_result
    tool_observations = _derive_tool_observations(state)
    final_answer = _compose_final_answer(grounded, reflection)
    if not reflection_used_fallback:
        trace_llm_decision(
            trace,
            phase="reflection",
            action="accept",
            accepted=True,
            reason="reflection_reviewed",
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


@traced_span(
    "agent.interaction",
    lambda args: {
        "model": args["context"].model_name,
        "prompt_length": len(args["user_prompt"]),
    },
)
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
        steps=[],
        request_analysis=analyze_request(user_prompt, context.tool_names),
        fulfillment={},
    )
    state.fulfillment = _initialize_fulfillment(state.request_analysis)
    draft_answer = ""
    duplicate_skip_counts: Dict[str, int] = {}

    for step_number in range(1, MAX_REACT_STEPS + 1):
        planner_messages = _render_planner_messages(state)
        react_messages = build_react_messages(
            planner_messages,
            context.tool_names,
            step_number=step_number,
            max_steps=MAX_REACT_STEPS,
            request_analysis=state.request_analysis,
        )
        try:
            decision = llm_react_json(
                react_messages,
                context.model_name,
                allowed_tools=context.tool_names,
                trace=trace,
                prompt_contract=REACT_PROMPT_CONTRACT,
            )
            if not isinstance(decision, ReactDecision):
                decision = validate_react_decision(decision)
            validate_react_decision_semantics(decision, context.tool_names)
        except Exception as exc:
            logger.exception("ReAct decision failed: %s", exc)
            if _derive_tool_observations(state):
                return finalize_after_execution(
                    context,
                    user_prompt,
                    state,
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
            pending_categories = _pending_requested_categories(state)
            if pending_categories:
                trace_llm_decision(
                    trace,
                    phase="react",
                    step_number=step_number,
                    action="finish",
                    accepted=False,
                    reason="pending_requested_work",
                )
                logger.info(
                    "Finish blocked: requested work is still pending for %s | event=react.finish_blocked reason=pending_requested_work pending=%s",
                    ", ".join(_category_label(tool_name) for tool_name in pending_categories),
                    ",".join(pending_categories),
                )
                state.steps.append(
                    ExecutionStep(
                        kind="finish_blocked",
                        thought=decision.thought,
                        outcome="pending_requested_work",
                        final_answer=decision.final_answer or "",
                        feedback_message=(
                            "Finish blocked: requested work is still pending for "
                            + ", ".join(_category_label(tool_name) for tool_name in pending_categories)
                            + ". Fetch the remaining requested categories or surface a supported failure first."
                        ),
                    )
                )
                continue
            trace_llm_decision(
                trace,
                phase="react",
                step_number=step_number,
                action="finish",
                accepted=True,
                reason="finish_selected",
            )
            draft_answer = decision.final_answer or ""
            state.final_answer = draft_answer
            state.steps.append(
                ExecutionStep(
                    kind="finish",
                    thought=decision.thought,
                    final_answer=draft_answer,
                    outcome="completed",
                )
            )
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
            trace_llm_decision(
                trace,
                phase="react",
                step_number=step_number,
                action="tool",
                tool_name=decision.tool,
                accepted=False,
                reason="invalid_tool_args",
            )
            parsed_payload = parse_tool_payload(
                decision.tool,
                {"error": f"{decision.tool} failed", "details": error or "invalid args"},
            )
            state.steps.append(
                ExecutionStep(
                    kind="tool_invalid",
                    thought=decision.thought,
                    tool_name=decision.tool,
                    raw_args=decision.args,
                    parsed_payload=parsed_payload,
                    outcome="invalid_args",
                    feedback_message=_planner_tool_feedback_detail(
                        decision.tool, decision.args, parsed_payload
                    ),
                )
            )
            _update_fulfillment(state, decision.tool, parsed_payload)
        elif has_successful_duplicate_observation(
            state.steps, decision.tool, normalized_args
        ):
            trace_llm_decision(
                trace,
                phase="react",
                step_number=step_number,
                action="tool",
                tool_name=decision.tool,
                accepted=False,
                reason="duplicate_successful_tool_call",
            )
            logger.info(
                "Skipping duplicate successful tool call for %s with args=%s and continuing",
                decision.tool,
                dump_tool_args(normalized_args),
            )
            if trace is not None:
                trace.add_event(
                    "duplicate_tool_call_skipped",
                    step_number=step_number,
                    tool_name=decision.tool,
                    args=dump_tool_args(normalized_args),
                    decision_summary=decision.thought,
                )
            signature = _planner_message_signature(decision.tool, normalized_args)
            duplicate_skip_counts[signature] = duplicate_skip_counts.get(signature, 0) + 1
            state.steps.append(
                ExecutionStep(
                    kind="tool_skip",
                    thought=decision.thought,
                    tool_name=decision.tool,
                    raw_args=decision.args,
                    normalized_args=normalized_args,
                    outcome="duplicate_skipped",
                    feedback_message=(
                        f"- {decision.tool}: duplicate successful call already exists for "
                        f"{_format_observation_args(normalized_args)}. "
                        "This action is exhausted; choose a different needed step or finish."
                    ),
                )
            )
            if duplicate_skip_counts[signature] >= 2:
                logger.info(
                    "Planner stuck on repeated duplicate successful tool call for %s with args=%s; finalizing early",
                    decision.tool,
                    dump_tool_args(normalized_args),
                )
                state.used_fallback = True
                break
            continue
        else:
            trace_llm_decision(
                trace,
                phase="react",
                step_number=step_number,
                action="tool",
                tool_name=decision.tool,
                accepted=True,
                reason="tool_selected",
            )
            duplicate_skip_counts[_planner_message_signature(decision.tool, normalized_args)] = 0
            raw_payload = await execute_tool_call(
                tool_gateway,
                decision.tool,
                dump_tool_args(normalized_args),
                step_number=step_number,
                trace=trace,
            )
            parsed_payload = parse_tool_payload(decision.tool, raw_payload)
            state.steps.append(
                ExecutionStep(
                    kind="tool_call",
                    thought=decision.thought,
                    tool_name=decision.tool,
                    raw_args=decision.args,
                    normalized_args=normalized_args,
                    parsed_payload=parsed_payload,
                    outcome="completed",
                    feedback_message=_planner_tool_feedback_detail(
                        decision.tool, normalized_args, parsed_payload
                    ),
                )
            )
            _update_fulfillment(state, decision.tool, parsed_payload)
    else:
        draft_answer = build_react_failure_answer()

    result = finalize_after_execution(
        context,
        user_prompt,
        state,
        draft_answer,
        used_fallback=state.used_fallback,
        trace=trace,
    )
    logger.info(
        "Interaction completed with %d observations and answer length %d",
        len(result.tool_observations),
        len(result.answer),
    )
    return result
