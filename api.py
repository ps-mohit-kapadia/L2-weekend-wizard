from __future__ import annotations

"""HTTP API interface for Weekend Wizard."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import requests
import uvicorn

from application.service import WeekendWizardApp
from config.config import get_settings
from llm_client import discover_model
from logger.logging import get_logger
from logger.tracing.request_trace import create_trace, render_trace
from schemas.api import ChatRequest, ChatResponse, HealthResponse, ReadinessChecks, ReadinessResponse


logger = get_logger("agent.api")

UNEXPECTED_CHAT_ERROR_DETAIL = "Weekend Wizard could not complete that request."
UNEXPECTED_READINESS_ERROR_DETAIL = "Weekend Wizard runtime failed to start."
OLLAMA_UNREACHABLE_DETAIL = "Ollama is not reachable."
MODEL_UNAVAILABLE_DETAIL = "Resolved model is not available in Ollama."
MCP_SERVER_MISSING_DETAIL = "MCP server file is missing."
STARTING_READINESS_DETAIL = "API runtime is starting."


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
        details = UNEXPECTED_READINESS_ERROR_DETAIL
        if isinstance(exc, requests.RequestException):
            details = OLLAMA_UNREACHABLE_DETAIL
        elif str(exc).startswith("Could not reach Ollama"):
            details = OLLAMA_UNREACHABLE_DETAIL
        elif str(exc).startswith("Configured Ollama model is not available"):
            details = MODEL_UNAVAILABLE_DETAIL

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
    app.state.startup_task = asyncio.create_task(warm_runtime(app, server_path))
    try:
        yield
    finally:
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

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        """Run one Weekend Wizard interaction through the shared app service.

        Args:
            request: API request payload describing the user prompt.

        Returns:
            The final structured chat response.

        Raises:
            HTTPException: If startup or interaction execution fails.
        """
        trace = create_trace(request.prompt)
        wizard = getattr(app.state, "wizard", None)
        readiness = app.state.readiness
        try:
            if readiness.status != "ready" or wizard is None or not wizard.is_initialized:
                logger.warning("Rejecting chat request because runtime is not ready: %s", readiness.details)
                raise HTTPException(status_code=503, detail=readiness.details or "Service is not ready.")

            logger.info(
                "Received /chat request with prompt length %d",
                len(request.prompt),
            )
            context = wizard.create_interaction_context()
            result = await wizard.run_interaction(request.prompt, context=context, trace=trace)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Chat request failed: %s", exc)
            raise HTTPException(status_code=500, detail=UNEXPECTED_CHAT_ERROR_DETAIL) from exc
        finally:
            if not trace.events or trace.events[-1].event != "interaction_completed":
                trace.add_event("interaction_completed")
            logger.info(render_trace(trace))

        logger.info(
            "Completed /chat request with %d observations, fallback=%s, answer length=%d",
            len(result.tool_observations),
            result.used_fallback,
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
