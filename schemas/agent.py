from __future__ import annotations

"""Typed models for planning, orchestration state, and interaction results."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, TypeAdapter


class PlanLocation(BaseModel):
    """Location details inferred by the planner."""

    city: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class PlanStep(BaseModel):
    """One executable tool step produced by the planner."""

    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)


class ExecutionPlan(BaseModel):
    """Planner output for one Weekend Wizard interaction."""

    goal: Literal["weekend_plan", "weather_lookup", "book_suggestions", "joke", "dog_photo", "trivia"]
    location: Optional[PlanLocation] = None
    book_topic: Optional[str] = None
    requested_tools: List[str] = Field(default_factory=list)
    execution_steps: List[PlanStep] = Field(default_factory=list)


class ReflectionResult(BaseModel):
    """One-shot reflection payload used to revise the final grounded answer."""

    answer: str


class ToolObservation(BaseModel):
    """Structured record of one tool invocation and its payload."""

    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    payload: str


class OrchestratorContext(BaseModel):
    """Runtime state required to process one user interaction."""

    tool_names: List[str] = Field(default_factory=list)
    history: List[Dict[str, str]] = Field(default_factory=list)
    model_name: str


class InteractionResult(BaseModel):
    """Structured result returned by the orchestrator for one interaction."""

    answer: str
    tool_observations: List[ToolObservation] = Field(default_factory=list)
    used_fallback: bool = False


_execution_plan_adapter = TypeAdapter(ExecutionPlan)
_reflection_result_adapter = TypeAdapter(ReflectionResult)


def validate_execution_plan(payload: Dict[str, Any]) -> ExecutionPlan:
    """Validate raw model JSON into a typed execution plan."""
    return _execution_plan_adapter.validate_python(payload)


def validate_reflection_result(payload: Dict[str, Any]) -> ReflectionResult:
    """Validate raw model JSON into a typed reflection payload."""
    return _reflection_result_adapter.validate_python(payload)
