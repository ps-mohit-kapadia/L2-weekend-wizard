from __future__ import annotations

"""ReAct-style tool orchestration for one Weekend Wizard interaction."""

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from agent.grounding import (
    build_grounded_items,
    compose_grounded_answer_from_observations,
    parse_tool_observations,
    parse_tool_payload_text,
    render_compact_observation_summaries,
)
from agent.policies.guardrails import (
    RequestAnalysis,
    analyze_request,
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
from schemas.tools import BookResults, DogResult, JokeResult, ToolError, TriviaResult, WeatherResult


logger = get_logger("agent.orchestrator")
MAX_REACT_STEPS = 6
SAFE_TOOL_INVOCATION_DETAIL = "tool execution failed"


@dataclass
class ExecutionState:
    """Mutable execution state for one Weekend Wizard interaction."""

    user_prompt: str
    tool_observations: List[ToolObservation]
    request_analysis: RequestAnalysis | None = None


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
    """Render tool args compactly for planner-visible assistant/tool messages."""
    if not args:
        return "{}"
    return json.dumps(args, sort_keys=True, separators=(",", ":"))


def _planner_message_signature(tool_name: str, args: Dict[str, Any]) -> str:
    return f"{tool_name}:{_format_observation_args(args)}"


def _planner_tool_feedback_detail(
    tool_name: str,
    args: Dict[str, Any],
    parsed: Any,
) -> str:
    """Render planner-local tool feedback without reusing grounding summaries."""
    if isinstance(parsed, ToolError):
        detail = parsed.details or parsed.error
        if tool_name == "get_weather":
            latitude = args.get("latitude")
            longitude = args.get("longitude")
            if latitude is not None and longitude is not None:
                return f"- Weather: {latitude}, {longitude} unavailable ({detail})"
            return f"- Weather: requested location unavailable ({detail})"
        if tool_name == "city_to_coords":
            return f"- City Lookup: unavailable ({detail})"
        if tool_name == "book_recs":
            return f"- Books: unavailable ({detail})"
        if tool_name == "random_joke":
            return f"- Joke: unavailable ({detail})"
        if tool_name == "random_dog":
            return f"- Dog Pic: unavailable ({detail})"
        if tool_name == "trivia":
            return f"- Trivia: unavailable ({detail})"
        return f"- {tool_name}: failed ({detail})"

    if tool_name == "city_to_coords":
        city = getattr(parsed, "city", None)
        latitude = getattr(parsed, "latitude", None)
        longitude = getattr(parsed, "longitude", None)
        if city is not None and latitude is not None and longitude is not None:
            return f"- City Lookup: {city}: {latitude}, {longitude}"
    elif tool_name == "get_weather":
        temperature = getattr(parsed, "temperature", None)
        if temperature is not None:
            latitude = args.get("latitude")
            longitude = args.get("longitude")
            label = (
                f"{latitude}, {longitude}"
                if latitude is not None and longitude is not None
                else "requested location"
            )
            unit = getattr(parsed, "temperature_unit", "") or ""
            summary = getattr(parsed, "weather_summary", None) or "current conditions"
            return f"- Weather: {label}: {temperature}{unit}, {summary}"
    elif tool_name == "book_recs":
        topic = getattr(parsed, "topic", None)
        results = getattr(parsed, "results", None) or []
        titles = [
            f"{book.title} by {book.author}"
            for book in results[:2]
            if getattr(book, "title", None)
        ]
        if titles:
            return f"- Books: {'; '.join(titles)}"
        if topic:
            return f"- Books: fetched results for {topic}"
    elif tool_name == "random_joke":
        joke = getattr(parsed, "joke", None)
        if joke:
            return f"- Joke: {joke}"
    elif tool_name == "random_dog":
        return "- Dog Pic: fetched one dog image"
    elif tool_name == "trivia":
        question = getattr(parsed, "question", None)
        correct_answer = getattr(parsed, "correct_answer", None)
        incorrect_answers = getattr(parsed, "incorrect_answers", None) or []
        if question and correct_answer:
            choices = incorrect_answers + [correct_answer]
            return f"- Trivia: {question} Choices: {', '.join(choices)}"

    return f"- {tool_name}: completed"


def _tool_feedback_from_payload(
    tool_name: str,
    args: Dict[str, Any],
    payload: str,
) -> str:
    parsed = parse_tool_payload_text(tool_name, payload)
    return _planner_tool_feedback_detail(tool_name, args, parsed)


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
        city = (
            args.get("city")
            or (state.request_analysis.city if state.request_analysis is not None else None)
            or infer_city(state.user_prompt)
        )
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
            or (state.request_analysis.book_topic if state.request_analysis is not None else None)
            or infer_book_topic(state.user_prompt)
        )
        limit = (
            args.get("limit")
            or (state.request_analysis.book_limit if state.request_analysis is not None else None)
            or infer_book_limit(state.user_prompt)
        )
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
    grounded = compose_grounded_answer_from_observations(
        user_prompt, "", tool_observations
    )
    if grounded.strip() or not tool_observations:
        return grounded

    compact_items = render_compact_observation_summaries(
        user_prompt, tool_observations
    )
    if compact_items:
        return "Weekend Wizard Results\n" + "\n".join(compact_items)
    return grounded


def _normalize_answer_text(text: str) -> str:
    return " ".join(text.lower().split())


def _reflection_preserves_grounded_content(
    user_prompt: str,
    tool_observations: List[ToolObservation],
    grounded: str,
    reflected: str,
) -> bool:
    """Return whether reflection preserved the grounded answer's required facts."""
    reflected_normalized = _normalize_answer_text(reflected)
    if not reflected_normalized:
        return False

    grounded_lines = [line.strip() for line in grounded.splitlines() if line.strip()]
    if not grounded_lines or not tool_observations:
        return True

    parsed_observations = parse_tool_observations(tool_observations)
    successful_weather = 0

    for observation in parsed_observations:
        payload = observation.payload
        if observation.tool_name == "city_to_coords":
            if isinstance(payload, ToolError):
                if "unavailable" not in reflected_normalized:
                    return False
            continue

        if observation.tool_name == "get_weather":
            if isinstance(payload, ToolError):
                if "unavailable" not in reflected_normalized:
                    return False
                if SAFE_TOOL_INVOCATION_DETAIL in _normalize_answer_text(payload.details or ""):
                    if SAFE_TOOL_INVOCATION_DETAIL not in reflected_normalized:
                        return False
                continue
            if isinstance(payload, WeatherResult) and payload.temperature is not None:
                successful_weather += 1
                temperature_fragment = _normalize_answer_text(
                    f"{payload.temperature}{payload.temperature_unit or ''}"
                )
                if temperature_fragment not in reflected_normalized:
                    return False
                summary_fragment = _normalize_answer_text(payload.weather_summary or "current conditions")
                if summary_fragment and summary_fragment not in reflected_normalized:
                    return False
                continue

        if observation.tool_name == "book_recs":
            if isinstance(payload, ToolError):
                if "unavailable" not in reflected_normalized:
                    return False
                continue
            if isinstance(payload, BookResults) and payload.results:
                for book in payload.results[:2]:
                    if book.title and _normalize_answer_text(book.title) not in reflected_normalized:
                        return False
                continue

        if observation.tool_name == "random_joke":
            if isinstance(payload, ToolError):
                if "unavailable" not in reflected_normalized:
                    return False
                continue
            if isinstance(payload, JokeResult):
                if _normalize_answer_text(payload.joke) not in reflected_normalized:
                    return False
                continue

        if observation.tool_name == "random_dog":
            if isinstance(payload, ToolError):
                if "unavailable" not in reflected_normalized:
                    return False
                continue
            if isinstance(payload, DogResult):
                if _normalize_answer_text(payload.image_url) not in reflected_normalized:
                    return False
                continue

        if observation.tool_name == "trivia":
            if isinstance(payload, ToolError):
                if "unavailable" not in reflected_normalized:
                    return False
                continue
            if isinstance(payload, TriviaResult):
                if _normalize_answer_text(payload.question) not in reflected_normalized:
                    return False
                continue

    grounded_items = build_grounded_items(user_prompt, parsed_observations)
    for item in grounded_items:
        if item.title == "City Lookup" and successful_weather:
            continue
        normalized_title = _normalize_answer_text(item.title)
        normalized_detail = _normalize_answer_text(item.detail)
        if "unavailable" in normalized_detail and "unavailable" not in reflected_normalized:
            return False
        if (
            SAFE_TOOL_INVOCATION_DETAIL in normalized_detail
            and SAFE_TOOL_INVOCATION_DETAIL not in reflected_normalized
        ):
            return False
    return True


def run_reflection(
    context: OrchestratorContext,
    user_prompt: str,
    tool_observations: List[ToolObservation],
    draft_answer: str,
    *,
    trace: RequestTrace | None = None,
) -> Tuple[str, bool]:
    """Run one quality pass over the grounded draft and fall back on failure."""
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
    """Build grounded output first, then run one quality-only reflection pass.

    The grounded draft is the semantic source for tool-backed turns. Reflection
    is allowed to improve presentation quality, but grounded output remains the
    authority baseline when reflection fails or drifts away from grounded facts.
    """
    grounded = (
        build_grounded_draft(user_prompt, tool_observations)
        if tool_observations
        else draft_answer
    )
    final_answer, reflection_used_fallback = run_reflection(
        context, user_prompt, tool_observations, grounded, trace=trace
    )
    if tool_observations and not reflection_used_fallback:
        if not _reflection_preserves_grounded_content(
            user_prompt,
            tool_observations,
            grounded,
            final_answer,
        ):
            logger.info(
                "Reflection drifted from grounded content; returning grounded draft instead"
            )
            final_answer = grounded
            reflection_used_fallback = True
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
    # `context.history` is the external conversation record. Planner continuity
    # now lives in a local assistant/tool transcript for this interaction only.
    planner_messages: List[Dict[str, str]] = [
        {"role": "user", "content": user_prompt}
    ]

    state = ExecutionState(
        user_prompt=user_prompt,
        tool_observations=[],
        request_analysis=analyze_request(user_prompt, context.tool_names),
    )
    # `state.tool_observations` remains the canonical execution truth used by
    # grounding and finalization. Planner messages are not a source of truth.
    draft_answer = ""
    used_fallback = False
    duplicate_skip_counts: Dict[str, int] = {}

    for step_number in range(1, MAX_REACT_STEPS + 1):
        react_messages = build_react_messages(
            planner_messages,
            context.tool_names,
            step_number=step_number,
            max_steps=MAX_REACT_STEPS,
            request_analysis=state.request_analysis,
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
        planner_args = normalized_args or decision.args
        planner_messages.append(
            {
                "role": "assistant",
                "content": (
                    f"Thought: {decision.thought}\n"
                    f"Action: tool\n"
                    f"Tool: {decision.tool}\n"
                    f"Args: {_format_observation_args(planner_args)}"
                ),
            }
        )
        if normalized_args is None:
            payload = _tool_error_payload(decision.tool, error or "invalid args")
            planner_messages.append(
                {
                    "role": "tool",
                    "content": _tool_feedback_from_payload(
                        decision.tool,
                        decision.args,
                        payload,
                    ),
                }
            )
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
            signature = _planner_message_signature(decision.tool, normalized_args)
            duplicate_skip_counts[signature] = duplicate_skip_counts.get(signature, 0) + 1
            planner_messages.append(
                {
                    "role": "tool",
                    "content": (
                        f"- {decision.tool}: duplicate successful call already exists for "
                        f"{_format_observation_args(normalized_args)}. "
                        "This action is exhausted; choose a different needed step or finish."
                    ),
                }
            )
            if duplicate_skip_counts[signature] >= 2:
                logger.info(
                    "Planner stuck on repeated duplicate successful tool call for %s with args=%s; finalizing early",
                    decision.tool,
                    normalized_args,
                )
                used_fallback = True
                break
            continue
        else:
            duplicate_skip_counts[_planner_message_signature(decision.tool, normalized_args)] = 0
            payload = await execute_tool_call(
                tool_gateway,
                decision.tool,
                normalized_args,
                step_number=step_number,
                trace=trace,
            )
            planner_messages.append(
                {
                    "role": "tool",
                    "content": _tool_feedback_from_payload(
                        decision.tool,
                        normalized_args,
                        payload,
                    ),
                }
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
        used_fallback=used_fallback,
        trace=trace,
    )
    logger.info(
        "Interaction completed with %d observations and answer length %d",
        len(result.tool_observations),
        len(result.answer),
    )
    return result
