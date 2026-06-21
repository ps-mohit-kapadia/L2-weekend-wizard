from __future__ import annotations

"""Typed models for ReAct decisions, orchestration state, and interaction results."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, TypeAdapter


class ReactDecision(BaseModel):
    """One bounded ReAct decision produced by the local model."""

    thought: str
    action: Literal["tool", "finish"]
    tool: Optional[str] = None
    args: Dict[str, Any] = Field(default_factory=dict)
    final_answer: Optional[str] = None


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


_react_decision_adapter = TypeAdapter(ReactDecision)
_reflection_result_adapter = TypeAdapter(ReflectionResult)


def validate_react_decision(payload: Dict[str, Any]) -> ReactDecision:
    """Validate raw model JSON into a typed ReAct decision."""
    decision = _react_decision_adapter.validate_python(payload)
    if decision.action == "tool" and not decision.tool:
        raise ValueError("Tool actions must include a tool name.")
    if decision.action == "finish" and not decision.final_answer:
        raise ValueError("Finish actions must include a final_answer.")
    return decision


def validate_reflection_result(payload: Dict[str, Any]) -> ReflectionResult:
    """Validate raw model JSON into a typed reflection payload."""
    return _reflection_result_adapter.validate_python(payload)
