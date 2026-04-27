from __future__ import annotations

"""Typed request and response models for the Weekend Wizard HTTP API."""

from typing import List

from pydantic import BaseModel, Field

from schemas.agent import ToolObservation


class ChatRequest(BaseModel):
    """Input payload for the Weekend Wizard chat endpoint.

    Attributes:
        prompt: User request to process.
    """

    prompt: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """Output payload returned by the Weekend Wizard chat endpoint.

    Attributes:
        answer: Final grounded answer generated for the request.
        tool_observations: Structured tool observations collected during execution.
        used_fallback: Whether the response came from a degraded fallback path.
    """

    answer: str
    tool_observations: List[ToolObservation] = Field(default_factory=list)
    used_fallback: bool = False


class HealthResponse(BaseModel):
    """Simple health payload returned by the HTTP API.

    Attributes:
        status: Health status for the API process.
    """

    status: str


class ReadinessChecks(BaseModel):
    """Structured readiness checks for the Weekend Wizard application.

    Attributes:
        model_resolved: Whether an Ollama model name was resolved successfully.
        model_available: Whether the resolved model is currently available in Ollama.
        server_path_exists: Whether the configured MCP server entrypoint exists.
        ollama_reachable: Whether the local Ollama runtime is reachable.
        mcp_session_ready: Whether an MCP session can be started successfully.
        tools_discovered: Whether MCP tool discovery returned at least one tool.
    """

    model_resolved: bool
    model_available: bool
    server_path_exists: bool
    ollama_reachable: bool
    mcp_session_ready: bool
    tools_discovered: bool


class ReadinessResponse(BaseModel):
    """Detailed readiness payload returned by the HTTP API.

    Attributes:
        status: Overall readiness state for the application.
        model_name: The resolved model name used for request handling.
        tool_count: Number of MCP tools discovered during readiness checks.
        checks: Structured readiness check results.
        details: Optional failure details when the service is not ready.
    """

    status: str
    model_name: str
    tool_count: int
    checks: ReadinessChecks
    details: str | None = None
