from __future__ import annotations

"""Workflow orchestration for one Weekend Wizard interaction."""

import json
from typing import Any, Dict, List, Optional, Tuple

from config.config import get_settings
from agent.policies.guardrails import RequestAnalysis, analyze_request, missing_requested_tools
from agent.grounding import (
    compose_grounded_answer_from_observations,
    parse_tool_payload_text,
    parse_tool_observations,
)
from logger.logging import get_logger
from llm_client import llm_json
from mcp_runtime.client import ToolGateway, ToolInvocationError
from schemas.agent import (
    FinalAction,
    InteractionResult,
    OrchestratorContext,
    ToolObservation,
    validate_agent_decision,
)
from schemas.tools import GeoResult, ToolError


logger = get_logger("agent.orchestrator")

_TOOL_LABELS = {
    "get_weather": "weather",
    "book_recs": "book ideas",
    "random_joke": "a joke",
    "random_dog": "a dog photo",
    "trivia": "trivia",
}


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


def has_pending_requested_tools(
    user_prompt: str,
    tool_observations: List[ToolObservation],
) -> bool:
    """Check whether the user's requested tool categories are still incomplete.

    Args:
        user_prompt: The active user request being processed.
        tool_observations: Structured tool observations collected during the current interaction.

    Returns:
        True when one or more requested tool categories have not been satisfied yet.
    """
    known_payloads = parse_tool_observations(tool_observations)
    pending_tools = missing_requested_tools(user_prompt, known_payloads)
    return bool(pending_tools)


def append_missing_tool_guidance(
    user_prompt: str,
    history: List[Dict[str, str]],
    tool_observations: List[ToolObservation],
) -> List[str]:
    """Append an LLM-facing reminder when the model tries to finish too early.

    Args:
        user_prompt: The active user request being processed.
        history: Conversation history including prior tool observations.
        tool_observations: Structured tool observations collected during the current interaction.

    Returns:
        The requested tool names that are still missing from the interaction.
    """
    missing_tools = missing_requested_tools(user_prompt, parse_tool_observations(tool_observations))
    if not missing_tools:
        return []

    history.append(
        {
            "role": "system",
            "content": (
                "You have not yet satisfied all requested tool categories. "
                f"Missing tools: {', '.join(missing_tools)}. "
                "Return valid JSON and either call exactly one missing tool next or finish only if the user no longer needs it."
            ),
        }
    )
    return missing_tools


def build_final_answer(
    user_prompt: str,
    draft_answer: str,
    tool_observations: List[ToolObservation],
) -> str:
    """Build the final user-facing answer from the model draft and tool results.

    Args:
        user_prompt: The active user request being answered.
        draft_answer: The model-produced draft answer before grounding.
        tool_observations: Structured tool observations collected during the interaction.

    Returns:
        The grounded final answer.
    """
    return compose_grounded_answer_from_observations(user_prompt, draft_answer, tool_observations)


def build_interaction_result(
    history: List[Dict[str, str]],
    answer: str,
    tool_observations: List[ToolObservation],
    *,
    used_step_limit_fallback: bool,
) -> InteractionResult:
    """Persist the final assistant answer and create the interaction result.

    Args:
        history: Conversation history that should receive the final assistant message.
        answer: The final assistant answer to persist.
        tool_observations: Structured tool observations collected during the interaction.
        used_step_limit_fallback: Whether the answer came from the step-limit fallback path.

    Returns:
        The structured result for the completed interaction.
    """
    history.append({"role": "assistant", "content": answer})
    return InteractionResult(
        answer=answer,
        tool_observations=tool_observations,
        used_step_limit_fallback=used_step_limit_fallback,
    )


async def execute_tool_call(
    tool_gateway: ToolGateway,
    tool_name: str,
    args: Dict[str, Any],
) -> str:
    """Invoke one MCP tool and serialize its response payload.

    Args:
        tool_gateway: Tool invocation gateway backed by the MCP service layer.
        tool_name: Name of the MCP tool to invoke.
        args: JSON-serializable tool arguments.

    Returns:
        The serialized tool payload text, including an error payload on operational tool-call failure.
    """
    try:
        logger.info("Invoking tool %s with args=%s", tool_name, args)
        result = await tool_gateway.call_tool(tool_name, args)
        payload = render_tool_result(result)
        logger.info("Tool %s completed", tool_name)
        return payload
    except ToolInvocationError as exc:
        logger.exception("Tool %s failed: %s", tool_name, exc)
        return json.dumps(
            {"error": f"tool call failed for {tool_name}", "details": str(exc)}
        )


def record_tool_observation(
    history: List[Dict[str, str]],
    tool_observations: List[ToolObservation],
    tool_name: str,
    args: Dict[str, Any],
    payload: str,
) -> None:
    """Record a tool observation in both free-form and structured interaction state.

    Args:
        history: Conversation history used by the agent loop.
        tool_observations: Structured tool observations used by the app surfaces and tests.
        tool_name: Name of the tool that was invoked.
        args: Tool arguments used for the invocation.
        payload: Serialized payload returned by the tool.
    """
    tool_observations.append(
        ToolObservation(
            tool_name=tool_name,
            args=args,
            payload=payload,
        )
    )
    history.append({"role": "assistant", "content": f"[tool:{tool_name}] {payload}"})


def append_missing_tool_note(
    answer: str,
    missing_tools: List[str],
) -> str:
    """Append a short note describing any requested tool categories that remain missing."""
    if not missing_tools:
        return answer
    missing_labels = ", ".join(_TOOL_LABELS.get(tool_name, tool_name) for tool_name in missing_tools)
    return (
        f"{answer}\n\n"
        "I fetched the information above, but I still could not get "
        f"{missing_labels}."
    )


async def execute_planned_tool(
    tool_gateway: ToolGateway,
    context: OrchestratorContext,
    tool_observations: List[ToolObservation],
    tool_name: str,
    args: Dict[str, Any],
) -> str:
    """Execute a planned deterministic tool step and record its observation."""
    payload = await execute_tool_call(tool_gateway, tool_name, args)
    record_tool_observation(
        context.history,
        tool_observations,
        tool_name,
        args,
        payload,
    )
    return payload


async def run_deterministic_flow(
    tool_gateway: ToolGateway,
    context: OrchestratorContext,
    user_prompt: str,
    analysis: RequestAnalysis,
) -> InteractionResult:
    """Execute the bounded Weekend Wizard flow without model-led step selection."""
    logger.info("Using deterministic orchestration for requested tools=%s", list(analysis.requested_tools))
    context.history.append({"role": "user", "content": user_prompt})
    tool_observations: List[ToolObservation] = []

    resolved_coords = analysis.coords
    if "get_weather" in analysis.requested_tools and analysis.needs_city_lookup:
        payload = await execute_planned_tool(
            tool_gateway,
            context,
            tool_observations,
            "city_to_coords",
            {"city": analysis.city},
        )
        resolved_coords = geo_payload_to_coords(payload)
        if resolved_coords is None:
            logger.warning("City lookup did not return usable coordinates for %s", analysis.city)

    if "get_weather" in analysis.requested_tools and resolved_coords is not None:
        latitude, longitude = resolved_coords
        await execute_planned_tool(
            tool_gateway,
            context,
            tool_observations,
            "get_weather",
            {"latitude": latitude, "longitude": longitude},
        )

    if "book_recs" in analysis.requested_tools:
        await execute_planned_tool(
            tool_gateway,
            context,
            tool_observations,
            "book_recs",
            {"topic": analysis.book_topic, "limit": analysis.book_limit},
        )

    if "random_joke" in analysis.requested_tools:
        await execute_planned_tool(
            tool_gateway,
            context,
            tool_observations,
            "random_joke",
            {},
        )

    if "random_dog" in analysis.requested_tools:
        await execute_planned_tool(
            tool_gateway,
            context,
            tool_observations,
            "random_dog",
            {},
        )

    if "trivia" in analysis.requested_tools:
        await execute_planned_tool(
            tool_gateway,
            context,
            tool_observations,
            "trivia",
            {},
        )

    answer = compose_grounded_answer_from_observations(user_prompt, "", tool_observations)
    missing_tools = missing_requested_tools(
        user_prompt,
        parse_tool_observations(tool_observations),
    )
    answer = append_missing_tool_note(answer, missing_tools)
    logger.info(
        "Deterministic orchestration completed with %d observations and answer length %d",
        len(tool_observations),
        len(answer),
    )
    return build_interaction_result(
        context.history,
        answer=answer,
        tool_observations=tool_observations,
        used_step_limit_fallback=False,
    )


def finalize_after_step_limit(
    context: OrchestratorContext,
    user_prompt: str,
    tool_observations: List[ToolObservation],
) -> InteractionResult:
    """Finalize an interaction after the decision loop exhausts its step budget.

    Args:
        context: Runtime orchestration context for the current interaction.
        user_prompt: The active user request being processed.
        tool_observations: Structured tool observations collected so far.

    Returns:
        The best available interaction result, either grounded from tool results or a fallback message.
    """
    missing_tools = missing_requested_tools(
        user_prompt,
        parse_tool_observations(tool_observations),
    )

    if tool_observations and not missing_tools:
        logger.warning(
            "Step limit reached after prompt length %d with %d observations; returning grounded fallback",
            len(user_prompt),
            len(tool_observations),
        )
        answer = compose_grounded_answer_from_observations(user_prompt, "", tool_observations)
        return build_interaction_result(
            context.history,
            answer=answer,
            tool_observations=tool_observations,
            used_step_limit_fallback=True,
        )

    if tool_observations:
        logger.warning(
            "Step limit reached after prompt length %d with %d observations; returning partial grounded fallback with missing tools=%s",
            len(user_prompt),
            len(tool_observations),
            missing_tools,
        )
        answer = compose_grounded_answer_from_observations(user_prompt, "", tool_observations)
        missing_labels = ", ".join(_TOOL_LABELS.get(tool_name, tool_name) for tool_name in missing_tools)
        answer = (
            f"{answer}\n\n"
            "I fetched the information above, but I ran out of steps before I could also get "
            f"{missing_labels}."
        )
        return build_interaction_result(
            context.history,
            answer=answer,
            tool_observations=tool_observations,
            used_step_limit_fallback=True,
        )

    fallback = "I hit my step limit, but I can try again if you want a simpler request."
    logger.warning(
        "Step limit reached after prompt length %d with %d observations; returning generic fallback",
        len(user_prompt),
        len(tool_observations),
    )
    return build_interaction_result(
        context.history,
        answer=fallback,
        tool_observations=tool_observations,
        used_step_limit_fallback=True,
    )


async def orchestrate_interaction(
    tool_gateway: ToolGateway,
    context: OrchestratorContext,
    user_prompt: str,
) -> InteractionResult:
    """Run one full agent interaction from prompt to structured result.

    Args:
        tool_gateway: Tool invocation gateway backed by the MCP service layer.
        context: Runtime orchestration context shared across the active session.
        user_prompt: The current user request to process.

    Returns:
        The structured result for the completed interaction.
    """
    logger.info("Starting interaction for prompt length %d", len(user_prompt))
    deterministic_request = analyze_request(user_prompt, context.tool_names)
    if deterministic_request is not None:
        return await run_deterministic_flow(
            tool_gateway,
            context,
            user_prompt,
            deterministic_request,
        )

    settings = get_settings()
    context.history.append({"role": "user", "content": user_prompt})
    tool_observations: List[ToolObservation] = []
    for step_number in range(1, settings.max_steps + 1):
        logger.info("Starting controller step %d of %d", step_number, settings.max_steps)
        raw_decision = llm_json(context.history, context.model_name)
        decision = validate_agent_decision(raw_decision)
        action = decision.action
        logger.info("Controller step %d selected action %s", step_number, action)

        if action == "final" and has_pending_requested_tools(user_prompt, tool_observations):
            missing_tools = append_missing_tool_guidance(user_prompt, context.history, tool_observations)
            logger.info("Final answer was rejected because these tools are still missing: %s", missing_tools)
            continue

        if action == "final":
            draft_answer = decision.answer.strip() if isinstance(decision, FinalAction) else ""
            answer = build_final_answer(user_prompt, draft_answer, tool_observations)
            logger.info(
                "Interaction completed normally with %d observations and answer length %d",
                len(tool_observations),
                len(answer),
            )
            return build_interaction_result(
                context.history,
                answer=answer,
                tool_observations=tool_observations,
                used_step_limit_fallback=False,
            )

        tool_name = action
        args = getattr(decision, "args", {}) or {}
        if tool_name not in context.tool_names:
            logger.warning("Model requested unknown tool %s", tool_name)
            context.history.append(
                {
                    "role": "assistant",
                    "content": f"[tool-error] Unknown tool requested: {tool_name}",
                }
            )
            continue

        payload = await execute_tool_call(tool_gateway, tool_name, args)
        record_tool_observation(
            context.history,
            tool_observations,
            tool_name,
            args,
            payload,
        )

    result = finalize_after_step_limit(context, user_prompt, tool_observations)
    logger.info(
        "Interaction completed with fallback=%s, %d observations, and answer length %d",
        result.used_step_limit_fallback,
        len(result.tool_observations),
        len(result.answer),
    )
    return result
