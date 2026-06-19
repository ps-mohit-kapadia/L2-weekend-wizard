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
        correlation_id: App-wide correlation id for this execution.
        answer: Final grounded answer generated for the request.
        tool_observations: Structured tool observations collected during execution.
    """

    correlation_id: str
    answer: str
    tool_observations: List[ToolObservation] = Field(default_factory=list)


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
        provider_reachable: Whether the configured LLM provider is reachable.
        ollama_reachable: Whether the local Ollama runtime is reachable.
        mcp_session_ready: Whether an MCP session can be started successfully.
        tools_discovered: Whether MCP tool discovery returned at least one tool.
        auth_configured: Whether API key authentication is configured.
        rate_limit_configured: Whether request rate limiting is configured.
        trace_logging_configured: Whether request trace logging is configured.
    """

    model_resolved: bool
    model_available: bool
    server_path_exists: bool
    provider_reachable: bool
    ollama_reachable: bool
    mcp_session_ready: bool
    tools_discovered: bool
    auth_configured: bool
    rate_limit_configured: bool
    trace_logging_configured: bool


class ReadinessResponse(BaseModel):
    """Detailed readiness payload returned by the HTTP API.

    Attributes:
        status: Overall readiness state for the application.
        provider: Configured LLM provider used for request handling.
        model_name: The resolved model name used for request handling.
        tool_count: Number of MCP tools discovered during readiness checks.
        request_timeout_seconds: Configured request timeout for agent calls.
        rate_limit_requests: Configured request limit per client window.
        rate_limit_window_seconds: Configured rate-limit window size.
        checks: Structured readiness check results.
        details: Optional failure details when the service is not ready.
    """

    status: str
    provider: str
    model_name: str
    tool_count: int
    request_timeout_seconds: float
    rate_limit_requests: int
    rate_limit_window_seconds: int
    checks: ReadinessChecks
    details: str | None = None
