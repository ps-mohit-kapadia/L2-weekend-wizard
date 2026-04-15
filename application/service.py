from __future__ import annotations

"""Shared application service for Weekend Wizard interfaces."""

from pathlib import Path
from typing import Any, List, Sequence

from agent.orchestrator import orchestrate_interaction
from agent.prompts import build_system_prompt
from logger.context import ensure_request_context
from logger.logging import get_logger
from mcp_runtime.client import McpService
from schemas.agent import InteractionResult, OrchestratorContext


logger = get_logger("application.service", layer="application")


class WeekendWizardApp:
    """Manage shared application startup and interaction execution.

    Args:
        server_path: MCP server entrypoint to launch.
        model_name: Ollama model selected for the session.
        server_args: Optional extra arguments passed to the MCP server process.
    """

    def __init__(
        self,
        server_path: Path,
        model_name: str,
        server_args: Sequence[str] | None = None,
    ) -> None:
        self._server_path = server_path
        self._model_name = model_name
        self._server_args = list(server_args or [])
        self._mcp_service = McpService(server_path, server_args=self._server_args)
        self._listed_tools: List[Any] = []
        self._tool_names: List[str] = []
        self._system_history: List[dict[str, str]] = []

    @property
    def model_name(self) -> str:
        """Return the active Ollama model name for the session.

        Returns:
            The resolved model name used by the current app session.
        """
        return self._model_name

    @property
    def listed_tools(self) -> List[Any]:
        """Return MCP tool descriptors discovered during startup.

        Returns:
            The discovered MCP tool descriptors.
        """
        return self._listed_tools

    @property
    def tool_names(self) -> List[str]:
        """Return MCP tool names discovered during startup.

        Returns:
            Tool names exposed by the MCP server.
        """
        return self._tool_names

    @property
    def server_path(self) -> Path:
        """Return the MCP server entrypoint used by the application runtime.

        Returns:
            The configured MCP server entrypoint path.
        """
        return self._server_path

    @property
    def is_initialized(self) -> bool:
        """Return whether the shared runtime has been initialized successfully.

        Returns:
            True when the application runtime is ready for interactions.
        """
        return bool(self._system_history)

    async def __aenter__(self) -> WeekendWizardApp:
        """Initialize MCP resources and prepare interaction context.

        Returns:
            The initialized application service.

        Raises:
            RuntimeError: If startup validation fails.
        """
        logger.info(
            "app_session_starting",
            server=str(self._server_path),
            model=self._model_name,
            extra_args=self._server_args,
        )
        try:
            await self._mcp_service.__aenter__()
            self._listed_tools = self._mcp_service.tools
            self._tool_names = self._mcp_service.tool_names
            self._validate_startup()
            self._system_history = [
                {"role": "system", "content": build_system_prompt(self._listed_tools)}
            ]
            logger.info(
                "app_session_ready",
                model=self._model_name,
                tool_count=len(self._tool_names),
            )
            return self
        except Exception as exc:
            logger.exception("app_session_start_failed", details=str(exc))
            await self._mcp_service.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        """Release MCP resources held by the application service."""
        logger.info(
            "app_session_closing",
            model=self._model_name,
            tool_count=len(self._tool_names),
        )
        self._system_history = []
        self._listed_tools = []
        self._tool_names = []
        await self._mcp_service.__aexit__(exc_type, exc, exc_tb)

    def create_interaction_context(
        self,
        model_name: str | None = None,
    ) -> OrchestratorContext:
        """Create a fresh orchestration context for one interaction flow.

        Args:
            model_name: Optional model override for the interaction context.

        Returns:
            A new orchestration context with isolated conversation history.

        Raises:
            RuntimeError: If the application runtime has not been initialized yet.
        """
        if not self._system_history:
            raise RuntimeError("Weekend Wizard app has not been initialized.")

        return OrchestratorContext(
            tool_names=list(self._tool_names),
            history=[dict(message) for message in self._system_history],
            model_name=model_name or self._model_name,
        )

    async def run_interaction(
        self,
        user_prompt: str,
        *,
        context: OrchestratorContext,
    ) -> InteractionResult:
        """Run one user interaction through the shared orchestration flow.

        Args:
            user_prompt: The user request to process.
            context: Orchestration context to use for this interaction.

        Returns:
            The structured interaction result.

        Raises:
            RuntimeError: If the application service has not been initialized yet.
        """
        if not self._system_history:
            raise RuntimeError("Weekend Wizard app has not been initialized.")
        with ensure_request_context("app"):
            logger.info(
                "app_interaction_dispatch", model=context.model_name, prompt_length=len(user_prompt)
            )
            result = await orchestrate_interaction(
                self._mcp_service,
                context,
                user_prompt,
            )
            logger.info(
                "app_interaction_completed",
                model=context.model_name,
                observation_count=len(result.tool_observations),
                used_fallback=result.used_step_limit_fallback,
                answer_length=len(result.answer),
            )
            return result

    def _validate_startup(self) -> None:
        """Validate session readiness after MCP startup.

        Raises:
            RuntimeError: If no model is available or no MCP tools were discovered.
        """
        if not self._model_name.strip():
            raise RuntimeError("No Ollama model was resolved for this session.")
        if not self._tool_names:
            raise RuntimeError("Startup check could not discover any MCP tools.")
