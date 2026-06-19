from __future__ import annotations

"""HTTP API interface for Weekend Wizard."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import time
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import requests
import uvicorn

from application.service import WeekendWizardApp
from config.config import Settings, get_settings
from llm_client import discover_model
from logger.logging import get_logger
from logger.tracing.request_trace import create_trace, render_trace
from schemas.a2a import (
    A2AAgentCapabilities,
    A2AAgentCard,
    A2AAgentSkill,
    A2AArtifact,
    A2AJsonRpcError,
    A2AJsonRpcRequest,
    A2AJsonRpcResponse,
    A2AJsonRpcResult,
    A2APart,
    A2ATaskStatus,
)
from schemas.agent import InteractionResult
from schemas.api import ChatRequest, ChatResponse, HealthResponse, ReadinessChecks, ReadinessResponse


logger = get_logger("agent.api")

UNEXPECTED_CHAT_ERROR_DETAIL = "Weekend Wizard could not complete that request."
UNEXPECTED_READINESS_ERROR_DETAIL = "Weekend Wizard runtime failed to start."
OLLAMA_UNREACHABLE_DETAIL = "Ollama is not reachable."
MODEL_UNAVAILABLE_DETAIL = "Resolved model is not available in Ollama."
MCP_SERVER_MISSING_DETAIL = "MCP server file is missing."
STARTING_READINESS_DETAIL = "API runtime is starting."
STARTUP_RETRY_INTERVAL_SECONDS = 5.0
UNAUTHORIZED_DETAIL = "Invalid or missing API key."
PROMPT_TOO_LARGE_DETAIL = "Prompt is too large."
RATE_LIMITED_DETAIL = "Too many requests. Please try again later."
REQUEST_TIMEOUT_DETAIL = "Weekend Wizard request timed out."
A2A_UNSUPPORTED_METHOD_DETAIL = "Unsupported A2A method."
A2A_INVALID_REQUEST_DETAIL = "Invalid A2A JSON-RPC request."
A2A_INVALID_PARAMS_DETAIL = "Invalid A2A message params."


def a2a_error(request_id: str | int | None, code: int, message: str) -> JSONResponse:
    """Build one JSON-RPC error response for A2A protocol failures."""
    response = A2AJsonRpcResponse(
        id=request_id,
        error=A2AJsonRpcError(code=code, message=message),
    )
    return JSONResponse(content=response.model_dump(exclude_none=True))


def classify_startup_failure(exc: Exception) -> tuple[str, bool]:
    """Classify startup failures by user-facing detail and recovery policy."""
    if isinstance(exc, requests.RequestException):
        return OLLAMA_UNREACHABLE_DETAIL, True
    if str(exc).startswith("Could not reach Ollama"):
        return OLLAMA_UNREACHABLE_DETAIL, True
    if str(exc).startswith("Configured Ollama model is not available"):
        return MODEL_UNAVAILABLE_DETAIL, False
    return UNEXPECTED_READINESS_ERROR_DETAIL, False


def build_readiness_response(
    *,
    status: str,
    server_path: Path,
    model_name: str,
    wizard: WeekendWizardApp | None,
    details: str | None,
    provider_name: str,
    provider_reachable: bool,
    model_available: bool,
) -> ReadinessResponse:
    """Build the canonical readiness payload from authoritative lifecycle state."""
    tool_names = wizard.tool_names if wizard is not None else ()
    mcp_session_ready = wizard is not None and wizard.is_initialized
    tools_discovered = bool(tool_names)
    provider_uses_ollama = provider_name == "ollama"
    effective_server_path = wizard.server_path if wizard is not None else server_path

    return ReadinessResponse(
        status=status,
        model_name=model_name,
        tool_count=len(tool_names),
        checks=ReadinessChecks(
            model_resolved=bool(model_name.strip()),
            model_available=model_available if provider_uses_ollama else True,
            server_path_exists=effective_server_path.exists(),
            ollama_reachable=provider_reachable if provider_uses_ollama else True,
            mcp_session_ready=mcp_session_ready,
            tools_discovered=tools_discovered,
        ),
        details=details,
    )


def require_api_key(request: Request, api_key: str | None) -> None:
    """Require a matching API key when one is configured."""
    if api_key is None:
        return
    if request.headers.get("X-API-Key") != api_key:
        raise HTTPException(status_code=401, detail=UNAUTHORIZED_DETAIL)


def enforce_prompt_limit(prompt: str, max_chars: int) -> None:
    """Reject prompts that exceed the configured API boundary limit."""
    if len(prompt) > max_chars:
        raise HTTPException(status_code=413, detail=PROMPT_TOO_LARGE_DETAIL)


def _client_id(request: Request) -> str:
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",", maxsplit=1)[0].strip()
    return request.client.host if request.client is not None else "unknown"


def enforce_rate_limit(app: FastAPI, client_id: str, settings: Settings) -> None:
    """Apply a simple in-memory fixed-window rate limit for local/demo API use."""
    now = time.monotonic()
    window_start, count = app.state.rate_limits.get(client_id, (now, 0))
    if now - window_start >= settings.rate_limit_window_seconds:
        window_start, count = now, 0
    if count >= settings.rate_limit_requests:
        raise HTTPException(status_code=429, detail=RATE_LIMITED_DETAIL)
    app.state.rate_limits[client_id] = (window_start, count + 1)


def extract_a2a_prompt(params: dict[str, object] | None) -> str | None:
    """Extract the first text part from a minimal A2A message/send payload."""
    if not isinstance(params, dict):
        return None
    message = params.get("message")
    if not isinstance(message, dict):
        return None
    parts = message.get("parts")
    if not isinstance(parts, list):
        return None
    for part in parts:
        if isinstance(part, dict) and part.get("kind") == "text":
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                return text
    return None


async def run_agent_prompt(
    app: FastAPI,
    prompt: str,
    trace,
    settings: Settings,
) -> InteractionResult:
    """Run one prompt through the shared ready runtime."""
    wizard = getattr(app.state, "wizard", None)
    readiness = app.state.readiness
    if readiness.status != "ready" or wizard is None or not wizard.is_initialized:
        logger.warning("Rejecting agent request because runtime is not ready: %s", readiness.details)
        raise HTTPException(status_code=503, detail=readiness.details or "Service is not ready.")

    context = wizard.create_interaction_context()
    return await asyncio.wait_for(
        wizard.run_interaction(prompt, context=context, trace=trace),
        timeout=settings.request_timeout,
    )


async def close_wizard_if_present(wizard: WeekendWizardApp | None) -> None:
    """Close a runtime instance best-effort when one exists."""
    if wizard is not None:
        await wizard.__aexit__(None, None, None)


async def warm_runtime(app: FastAPI, server_path: Path) -> None:
    """Warm the shared runtime in the background and publish readiness state."""
    settings = get_settings()
    provider_name = settings.llm_provider
    model_name = ""
    wizard: WeekendWizardApp | None = None
    app.state.startup_retryable = False

    try:
        model_name = discover_model(None)
        app.state.readiness = build_readiness_response(
            status="not_ready",
            server_path=server_path,
            model_name=model_name,
            wizard=None,
            details=STARTING_READINESS_DETAIL,
            provider_name=provider_name,
            provider_reachable=False,
            model_available=False,
        )
        wizard = WeekendWizardApp(server_path, model_name, ["mcp-server"])
        await wizard.__aenter__()
        details: str | None = None
        runtime_server_path = wizard.server_path
        if not model_name.strip():
            details = "No Ollama model was resolved for this session."
        elif not runtime_server_path.exists():
            details = MCP_SERVER_MISSING_DETAIL
        elif not wizard.is_initialized:
            details = "Application runtime is not initialized."
        elif not wizard.tool_names:
            details = "Startup check could not discover any MCP tools."

        readiness = build_readiness_response(
            status="ready" if details is None else "not_ready",
            server_path=server_path,
            model_name=model_name,
            wizard=wizard,
            details=details,
            provider_name=provider_name,
            provider_reachable=True,
            model_available=True,
        )
        if readiness.status != "ready":
            logger.warning("API runtime is not ready: %s", readiness.details)
            app.state.readiness = readiness
            await close_wizard_if_present(wizard)
            return

        app.state.wizard = wizard
        app.state.readiness = readiness
        logger.info("API runtime ready with model %s and %d tools", wizard.model_name, len(wizard.tool_names))
    except asyncio.CancelledError:
        await close_wizard_if_present(wizard)
        raise
    except Exception as exc:
        logger.exception("API runtime startup failed: %s", exc)
        await close_wizard_if_present(wizard)
        provider_uses_ollama = provider_name == "ollama"
        details, retryable = classify_startup_failure(exc)
        app.state.startup_retryable = retryable

        app.state.readiness = build_readiness_response(
            status="not_ready",
            server_path=server_path,
            model_name=model_name,
            wizard=None,
            details=details,
            provider_name=provider_name,
            provider_reachable=not provider_uses_ollama or details != OLLAMA_UNREACHABLE_DETAIL,
            model_available=not provider_uses_ollama or details != MODEL_UNAVAILABLE_DETAIL,
        )


async def supervise_runtime(app: FastAPI, server_path: Path) -> None:
    """Own runtime startup and retry only failures that can recover in-process."""
    while not getattr(app.state, "startup_stopped", False):
        await warm_runtime(app, server_path)
        readiness = app.state.readiness
        if readiness.status == "ready" or not getattr(app.state, "startup_retryable", False):
            return
        logger.info("Retrying API runtime startup after retryable readiness failure: %s", readiness.details)
        await asyncio.sleep(STARTUP_RETRY_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage the shared Weekend Wizard runtime for the API process.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to FastAPI while the shared runtime is available.
    """
    project_dir = Path(__file__).resolve().parent
    server_path = project_dir / "main.py"
    provider_name = get_settings().llm_provider
    app.state.wizard = None
    app.state.startup_retryable = False
    app.state.startup_stopped = False
    app.state.rate_limits = {}
    app.state.readiness = build_readiness_response(
        status="not_ready",
        server_path=server_path,
        model_name="",
        wizard=None,
        details=STARTING_READINESS_DETAIL,
        provider_name=provider_name,
        provider_reachable=False,
        model_available=False,
    )
    app.state.startup_task = asyncio.create_task(supervise_runtime(app, server_path))
    try:
        yield
    finally:
        app.state.startup_stopped = True
        startup_task: asyncio.Task[None] | None = getattr(app.state, "startup_task", None)
        if startup_task is not None and not startup_task.done():
            startup_task.cancel()
            try:
                await startup_task
            except asyncio.CancelledError:
                pass
        wizard = getattr(app.state, "wizard", None)
        if wizard is not None:
            logger.info("Closing API runtime for model %s with %d tools", wizard.model_name, len(wizard.tool_names))
            await wizard.__aexit__(None, None, None)
        app.state.wizard = None
        app.state.startup_task = None


def create_api() -> FastAPI:
    """Create the FastAPI application for Weekend Wizard.

    Returns:
        A configured FastAPI application exposing health and chat routes.
    """
    app = FastAPI(title="Weekend Wizard API", version="1.0.0", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return a lightweight health signal for the API process."""
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=ReadinessResponse)
    async def ready() -> JSONResponse:
        """Return a readiness signal for the Weekend Wizard application."""
        response = app.state.readiness
        status_code = 200 if response.status == "ready" else 503
        return JSONResponse(status_code=status_code, content=response.model_dump())

    @app.get("/.well-known/agent.json", response_model=A2AAgentCard)
    async def agent_card(request: Request) -> A2AAgentCard:
        """Return the A2A Agent Card for Weekend Wizard discovery."""
        base_url = str(request.base_url).rstrip("/")
        return A2AAgentCard(
            name="Weekend Wizard",
            description="Grounded weekend-planning agent using weather, books, jokes, dog photos, and trivia.",
            url=f"{base_url}/a2a/jsonrpc",
            version="1.0.0",
            protocolVersion="0.3.0",
            capabilities=A2AAgentCapabilities(streaming=False),
            defaultInputModes=["text/plain"],
            defaultOutputModes=["text/plain"],
            skills=[
                A2AAgentSkill(
                    id="weekend_planning",
                    name="Weekend Planning",
                    description="Plan a weekend using supported public-data tools.",
                ),
                A2AAgentSkill(
                    id="weather",
                    name="Weather Lookup",
                    description="Fetch current weather by coordinates or city lookup.",
                ),
                A2AAgentSkill(
                    id="book_recommendations",
                    name="Book Recommendations",
                    description="Fetch book recommendations for a topic.",
                ),
                A2AAgentSkill(
                    id="entertainment",
                    name="Jokes, Dog Photos, and Trivia",
                    description="Fetch one safe joke, dog photo, or trivia question.",
                ),
            ],
        )

    @app.post("/a2a/jsonrpc", response_model=A2AJsonRpcResponse)
    async def a2a_jsonrpc(http_request: Request, request: A2AJsonRpcRequest) -> JSONResponse:
        """Handle minimal synchronous A2A JSON-RPC message/send requests."""
        if request.jsonrpc != "2.0":
            return a2a_error(request.id, -32600, A2A_INVALID_REQUEST_DETAIL)
        if request.method != "message/send":
            return a2a_error(request.id, -32601, A2A_UNSUPPORTED_METHOD_DETAIL)

        prompt = extract_a2a_prompt(request.params)
        if prompt is None:
            return a2a_error(request.id, -32602, A2A_INVALID_PARAMS_DETAIL)

        settings = get_settings()
        require_api_key(http_request, settings.api_key)
        enforce_prompt_limit(prompt, settings.max_prompt_chars)
        enforce_rate_limit(app, _client_id(http_request), settings)

        trace = create_trace(prompt)
        try:
            result = await run_agent_prompt(app, prompt, trace, settings)
        except asyncio.TimeoutError as exc:
            logger.warning("A2A request timed out after %ss", settings.request_timeout)
            raise HTTPException(status_code=504, detail=REQUEST_TIMEOUT_DETAIL) from exc
        except HTTPException as exc:
            return a2a_error(request.id, -32000, str(exc.detail))
        except Exception as exc:
            logger.exception("A2A request failed: %s", exc)
            return a2a_error(request.id, -32000, UNEXPECTED_CHAT_ERROR_DETAIL)
        finally:
            if not trace.events or trace.events[-1].event != "interaction_completed":
                trace.add_event("interaction_completed")
            logger.info(render_trace(trace))

        response = A2AJsonRpcResponse(
            id=request.id,
            result=A2AJsonRpcResult(
                status=A2ATaskStatus(state="completed"),
                artifacts=[
                    A2AArtifact(
                        name="weekend_wizard_answer",
                        parts=[A2APart(kind="text", text=result.answer)],
                    )
                ],
            ),
        )
        return JSONResponse(content=response.model_dump(exclude_none=True))

    @app.post("/chat", response_model=ChatResponse)
    async def chat(http_request: Request, request: ChatRequest) -> ChatResponse:
        """Run one Weekend Wizard interaction through the shared app service.

        Args:
            request: API request payload describing the user prompt.

        Returns:
            The final structured chat response.

        Raises:
            HTTPException: If startup or interaction execution fails.
        """
        settings = get_settings()
        require_api_key(http_request, settings.api_key)
        enforce_prompt_limit(request.prompt, settings.max_prompt_chars)
        enforce_rate_limit(app, _client_id(http_request), settings)

        trace = create_trace(request.prompt)
        try:
            logger.info(
                "Received /chat request with prompt length %d",
                len(request.prompt),
            )
            result = await run_agent_prompt(app, request.prompt, trace, settings)
        except HTTPException:
            raise
        except asyncio.TimeoutError as exc:
            logger.warning("Chat request timed out after %ss", settings.request_timeout)
            raise HTTPException(status_code=504, detail=REQUEST_TIMEOUT_DETAIL) from exc
        except Exception as exc:
            logger.exception("Chat request failed: %s", exc)
            raise HTTPException(status_code=500, detail=UNEXPECTED_CHAT_ERROR_DETAIL) from exc
        finally:
            if not trace.events or trace.events[-1].event != "interaction_completed":
                trace.add_event("interaction_completed")
            logger.info(render_trace(trace))

        logger.info(
            "Completed /chat request with %d observations, fallback=%s, answer length=%d | event=request.completed fallback=%s observations_count=%d answer_length=%d",
            len(result.tool_observations),
            result.used_fallback,
            len(result.answer),
            str(result.used_fallback).lower(),
            len(result.tool_observations),
            len(result.answer),
        )
        return ChatResponse(
            answer=result.answer,
            tool_observations=result.tool_observations,
        )

    return app


app = create_api()


def run_api() -> None:
    """Start the Weekend Wizard HTTP server using uvicorn."""
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
