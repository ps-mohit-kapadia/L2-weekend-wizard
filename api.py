from __future__ import annotations

"""HTTP API interface for Weekend Wizard."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import requests
import uvicorn

from application.service import WeekendWizardApp
from llm_client import discover_model
from llm_client import list_available_models
from logger.logging import get_logger
from schemas.api import ChatRequest, ChatResponse, HealthResponse, ReadinessChecks, ReadinessResponse


logger = get_logger("agent.api")


def build_not_ready_response(
    server_path: Path,
    model_name: str,
    details: str,
) -> ReadinessResponse:
    """Build a readiness payload for a failed or unavailable API runtime."""
    return ReadinessResponse(
        status="not_ready",
        model_name=model_name,
        tool_count=0,
        checks=ReadinessChecks(
            model_resolved=bool(model_name.strip()),
            model_available=False,
            server_path_exists=server_path.exists(),
            ollama_reachable=False,
            mcp_session_ready=False,
            tools_discovered=False,
        ),
        details=details,
    )


def evaluate_runtime_readiness(app: WeekendWizardApp) -> ReadinessResponse:
    """Evaluate whether an initialized API runtime is ready to serve."""
    checks = ReadinessChecks(
        model_resolved=bool(app.model_name.strip()),
        model_available=False,
        server_path_exists=app.server_path.exists(),
        ollama_reachable=False,
        mcp_session_ready=app.is_initialized,
        tools_discovered=bool(app.tool_names),
    )
    details: str | None = None

    if not checks.model_resolved:
        details = "No Ollama model was resolved for this session."
    elif not checks.server_path_exists:
        details = f"MCP server file not found: {app.server_path}"
    elif not checks.mcp_session_ready:
        details = "Application runtime is not initialized."
    elif not checks.tools_discovered:
        details = "Startup check could not discover any MCP tools."

    if details is None:
        try:
            available_models = list_available_models(timeout=5)
            checks.ollama_reachable = True
            checks.model_available = app.model_name in available_models
            if not checks.model_available:
                details = f"Resolved model is not available in Ollama: {app.model_name}"
        except requests.RequestException as exc:
            details = f"Ollama is not reachable: {exc}"

    status = "ready" if details is None and all(checks.model_dump().values()) else "not_ready"
    return ReadinessResponse(
        status=status,
        model_name=app.model_name,
        tool_count=len(app.tool_names),
        checks=checks,
        details=details,
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
    server_path, model_name = project_dir / "main.py", discover_model(None)
    wizard = WeekendWizardApp(server_path, model_name, ["mcp-server"])
    app.state.wizard = None
    app.state.readiness = build_not_ready_response(
        server_path,
        model_name,
        "API runtime has not started yet.",
    )

    try:
        await wizard.__aenter__()
    except Exception as exc:
        logger.exception("API runtime startup failed: %s", exc)
        app.state.readiness = build_not_ready_response(server_path, model_name, str(exc))
        yield
        return

    readiness = evaluate_runtime_readiness(wizard)
    if readiness.status != "ready":
        logger.warning("API runtime is not ready: %s", readiness.details)
        app.state.readiness = readiness
        await wizard.__aexit__(None, None, None)
        yield
        return

    app.state.wizard = wizard
    app.state.readiness = readiness
    logger.info("API runtime ready with model %s and %d tools", wizard.model_name, len(wizard.tool_names))
    try:
        yield
    finally:
        logger.info("Closing API runtime for model %s with %d tools", wizard.model_name, len(wizard.tool_names))
        await wizard.__aexit__(None, None, None)
        app.state.wizard = None


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
        wizard = getattr(app.state, "wizard", None)
        response = evaluate_runtime_readiness(wizard) if wizard is not None else app.state.readiness
        app.state.readiness = response

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
        wizard = getattr(app.state, "wizard", None)
        if wizard is None or not wizard.is_initialized:
            readiness = app.state.readiness
            logger.warning("Rejecting chat request because runtime is not ready: %s", readiness.details)
            raise HTTPException(status_code=503, detail=readiness.details or "Service is not ready.")

        logger.info(
            "Received /chat request with prompt length %d",
            len(request.prompt),
        )
        try:
            context = wizard.create_interaction_context()
            result = await wizard.run_interaction(request.prompt, context=context)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Chat request failed: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
