from __future__ import annotations

"""Workflow orchestration for one Weekend Wizard interaction."""

import json
from typing import Any, Dict, List

from config.config import get_settings
from agent.policies.guardrails import missing_requested_tools
from agent.grounding import (
    compose_grounded_answer_from_observations,
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


logger = get_logger("agent.orchestrator", layer="orchestrator")


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
        logger.info("tool_invocation", tool=tool_name, args=args)
        result = await tool_gateway.call_tool(tool_name, args)
        payload = render_tool_result(result)
        logger.info("tool_invocation_completed", tool=tool_name)
        return payload
    except ToolInvocationError as exc:
        logger.exception("tool_invocation_failed", tool=tool_name, details=str(exc))
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
    if not has_pending_requested_tools(user_prompt, tool_observations) and tool_observations:
        logger.warning(
            "interaction_step_limit_fallback",
            prompt_length=len(user_prompt),
            observation_count=len(tool_observations),
            mode="grounded",
        )
        answer = compose_grounded_answer_from_observations(user_prompt, "", tool_observations)
        return build_interaction_result(
            context.history,
            answer=answer,
            tool_observations=tool_observations,
            used_step_limit_fallback=True,
        )

    fallback = "I hit my step limit, but I can try again if you want a simpler request."
    logger.warning(
        "interaction_step_limit_fallback",
        prompt_length=len(user_prompt),
        observation_count=len(tool_observations),
        mode="generic",
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
    settings = get_settings()
    logger.info("interaction_start", prompt_length=len(user_prompt))
    context.history.append({"role": "user", "content": user_prompt})
    tool_observations: List[ToolObservation] = []

    for _ in range(settings.agent.max_steps):
        raw_decision = llm_json(context.history, context.model_name)
        decision = validate_agent_decision(raw_decision)
        action = decision.action
        logger.info("decision_received", action=action)

        if action == "final" and has_pending_requested_tools(user_prompt, tool_observations):
            missing_tools = append_missing_tool_guidance(user_prompt, context.history, tool_observations)
            logger.info("final_reprompted", reason="pending_requested_tools", missing_tools=missing_tools)
            continue

        if action == "final":
            draft_answer = decision.answer.strip() if isinstance(decision, FinalAction) else ""
            answer = build_final_answer(user_prompt, draft_answer, tool_observations)
            logger.info(
                "interaction_completed",
                used_fallback=False,
                observation_count=len(tool_observations),
                answer_length=len(answer),
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
            logger.warning("unknown_tool_requested", tool=tool_name)
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
        "interaction_completed",
        used_fallback=result.used_step_limit_fallback,
        observation_count=len(result.tool_observations),
        answer_length=len(result.answer),
    )
    return result
