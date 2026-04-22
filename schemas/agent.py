from __future__ import annotations

"""Typed models for planning, orchestration state, and interaction results."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, TypeAdapter


class PlanLocation(BaseModel):
    """Location details inferred by the planner.

    Attributes:
        city: Optional city name inferred from the prompt.
        latitude: Optional latitude associated with the request.
        longitude: Optional longitude associated with the request.
    """

    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class PlanStep(BaseModel):
    """One executable tool step produced by the planner.

    Attributes:
        tool: Tool name to invoke during deterministic execution.
        args: JSON-compatible arguments for the tool call.
    """

    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """Planner output for one Weekend Wizard interaction.

    Attributes:
        goal: High-level interaction goal selected by the planner.
        location: Optional location details inferred during planning.
        book_topic: Optional book topic used for recommendation requests.
        requested_tools: User-requested tool categories inferred by the planner.
        execution_steps: Ordered tool steps to execute deterministically.
    """

    goal: Literal["weekend_plan", "weather_lookup", "book_suggestions", "joke", "dog_photo", "trivia"]
    location: Optional[PlanLocation] = None
    book_topic: Optional[str] = None
    requested_tools: List[str] = Field(default_factory=list)
    execution_steps: List[PlanStep] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    """One-shot reflection payload used to revise the final grounded answer.

    Attributes:
        answer: Refined final answer returned by the reflection step.
    """

    answer: str


class ToolObservation(BaseModel):
    """Structured record of one tool invocation and its payload.

    Attributes:
        tool_name: Name of the tool that was invoked.
        args: JSON-compatible arguments used for the tool call.
        payload: Serialized payload returned by the tool.
    """

    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    payload: str


class OrchestratorContext(BaseModel):
    """Runtime state required to process one user interaction.

    Attributes:
        tool_names: Tool names available to the active runtime.
        history: Conversation history accumulated for the interaction.
        model_name: Active Ollama model name used for planner and reflection calls.
        request_id: Correlation identifier for one end-to-end request.
    """

    tool_names: List[str] = Field(default_factory=list)
    history: List[Dict[str, str]] = Field(default_factory=list)
    model_name: str
    request_id: str


class InteractionResult(BaseModel):
    """Structured result returned by the orchestrator for one interaction.

    Attributes:
        answer: Final answer returned to the caller.
        tool_observations: Structured tool outputs collected during execution.
        used_fallback: Whether the interaction completed through a fallback path.
    """

    answer: str
    tool_observations: List[ToolObservation] = Field(default_factory=list)
    used_fallback: bool = False


_execution_plan_adapter = TypeAdapter(ExecutionPlan)
_reflection_result_adapter = TypeAdapter(ReflectionResult)


def validate_execution_plan(payload: Dict[str, Any]) -> ExecutionPlan:
    """Validate raw model JSON into a typed execution plan.

    Args:
        payload: Raw JSON payload returned by the planner model.

    Returns:
        A validated execution plan.
    """
    return _execution_plan_adapter.validate_python(payload)


def validate_reflection_result(payload: Dict[str, Any]) -> ReflectionResult:
    """Validate raw model JSON into a typed reflection payload.

    Args:
        payload: Raw JSON payload returned by the reflection model.

    Returns:
        A validated reflection payload.
    """
    return _reflection_result_adapter.validate_python(payload)
