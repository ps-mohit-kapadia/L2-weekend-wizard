from __future__ import annotations

"""Typed A2A boundary models for Weekend Wizard."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class A2AAgentSkill(BaseModel):
    """One advertised A2A agent skill."""

    id: str
    name: str
    description: str


class A2AAgentCapabilities(BaseModel):
    """A2A capability flags supported by this adapter."""

    streaming: bool = False


class A2AAgentCard(BaseModel):
    """Public A2A Agent Card for discovery."""

    name: str
    description: str
    url: str
    version: str
    protocolVersion: str
    capabilities: A2AAgentCapabilities
    defaultInputModes: list[str]
    defaultOutputModes: list[str]
    skills: list[A2AAgentSkill]


class A2AJsonRpcRequest(BaseModel):
    """Minimal JSON-RPC request envelope accepted by the A2A adapter."""

    jsonrpc: str
    id: str | int | None = None
    method: str
    params: dict[str, Any] | None = None


class A2APart(BaseModel):
    """Text part used in A2A messages and artifacts."""

    kind: Literal["text"]
    text: str


class A2AArtifact(BaseModel):
    """A2A artifact returned from a completed task."""

    name: str
    parts: list[A2APart]


class A2ATaskStatus(BaseModel):
    """Minimal task status for synchronous A2A completion."""

    state: Literal["completed"]


class A2AJsonRpcResult(BaseModel):
    """Successful A2A JSON-RPC result payload."""

    status: A2ATaskStatus
    correlation_id: str | None = None
    artifacts: list[A2AArtifact] = Field(default_factory=list)


class A2AJsonRpcError(BaseModel):
    """JSON-RPC error object returned for A2A protocol failures."""

    code: int
    message: str


class A2AJsonRpcResponse(BaseModel):
    """JSON-RPC response envelope for A2A success or protocol errors."""

    jsonrpc: str = "2.0"
    id: str | int | None = None
    result: A2AJsonRpcResult | None = None
    error: A2AJsonRpcError | None = None
