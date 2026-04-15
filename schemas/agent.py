from __future__ import annotations

"""Typed models for agent decisions, orchestration state, and interaction results."""

from typing import Any, Dict, List, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, field_validator


class ToolAction(BaseModel):
    """Model decision instructing the agent to invoke a tool.

    Attributes:
        action: MCP tool name selected by the model.
        args: JSON-compatible arguments for the tool invocation.
    """

    action: str
    args: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("action")
    @classmethod
    def action_must_not_be_final(cls, value: str) -> str:
        if value == "final":
            raise ValueError("tool action cannot be 'final'")
        return value


class FinalAction(BaseModel):
    """Model decision instructing the agent to finish the interaction.

    Attributes:
        action: Literal marker indicating final-answer completion.
        answer: Draft answer proposed by the model.
    """

    action: Literal["final"]
    answer: str


class ToolObservation(BaseModel):
    """Structured record of one tool invocation and its payload.

    Attributes:
        tool_name: Name of the MCP tool that was invoked.
        args: Tool arguments used for the invocation.
        payload: Serialized payload returned by the tool.
    """

    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    payload: str


class OrchestratorContext(BaseModel):
    """Runtime state required to process one user interaction.

    Attributes:
        tool_names: Tool names available through the active MCP session.
        history: Conversation history used by the agent loop.
        model_name: Active Ollama model name for the current application session.
    """

    tool_names: List[str] = Field(default_factory=list)
    history: List[Dict[str, str]] = Field(default_factory=list)
    model_name: str


class InteractionResult(BaseModel):
    """Structured result returned by the orchestrator for one interaction.

    Attributes:
        answer: Final user-facing answer.
        tool_observations: Structured tool observations collected during the interaction.
        used_step_limit_fallback: Whether the answer came from the step-limit fallback path.
    """

    answer: str
    tool_observations: List[ToolObservation] = Field(default_factory=list)
    used_step_limit_fallback: bool = False


AgentDecision = Union[FinalAction, ToolAction]

_agent_decision_adapter = TypeAdapter(AgentDecision)


def validate_agent_decision(payload: Dict[str, Any]) -> AgentDecision:
    """Validate raw model JSON into a typed agent decision.

    Args:
        payload: Raw JSON payload returned by the model.

    Returns:
        A typed tool decision or final-answer decision.
    """
    return _agent_decision_adapter.validate_python(payload)
