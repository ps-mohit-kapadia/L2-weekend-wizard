from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from application.service import WeekendWizardApp
from schemas.agent import InteractionResult, OrchestratorContext


class _FakeMcpService:
    def __init__(self, *_args, **_kwargs) -> None:
        self.tools = ["weather"]
        self.tool_names = ["get_weather"]

    async def __aenter__(self) -> _FakeMcpService:
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        return None


class _NoToolsMcpService(_FakeMcpService):
    def __init__(self, *_args, **_kwargs) -> None:
        self.tools = []
        self.tool_names = []


class AppServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_app_service_initializes_context_and_tools(self) -> None:
        with (
            patch("application.service.McpService", _FakeMcpService),
            patch("application.service.build_system_prompt", return_value="system"),
        ):
            app = WeekendWizardApp(Path("main.py"), "llama3.2:latest", ["mcp-server"])
            await app.__aenter__()

        self.assertEqual(app.tool_names, ["get_weather"])
        self.assertEqual(app.model_name, "llama3.2:latest")
        self.assertTrue(app.is_initialized)

        await app.__aexit__(None, None, None)

    async def test_app_service_resets_initialization_state_on_close(self) -> None:
        with (
            patch("application.service.McpService", _FakeMcpService),
            patch("application.service.build_system_prompt", return_value="system"),
        ):
            app = WeekendWizardApp(Path("main.py"), "llama3.2:latest", ["mcp-server"])
            await app.__aenter__()
            await app.__aexit__(None, None, None)

        self.assertFalse(app.is_initialized)

    async def test_app_service_rejects_startup_without_tools(self) -> None:
        with patch("application.service.McpService", _NoToolsMcpService):
            app = WeekendWizardApp(Path("main.py"), "llama3.2:latest", ["mcp-server"])

            with self.assertRaises(RuntimeError):
                await app.__aenter__()

    async def test_run_interaction_uses_orchestrator(self) -> None:
        result = InteractionResult(answer="Ready", tool_observations=[], used_step_limit_fallback=False)

        with (
            patch("application.service.McpService", _FakeMcpService),
            patch("application.service.build_system_prompt", return_value="system"),
            patch("application.service.orchestrate_interaction", new=AsyncMock(return_value=result)) as mock_orchestrate,
        ):
            app = WeekendWizardApp(Path("main.py"), "llama3.2:latest", ["mcp-server"])
            await app.__aenter__()
            context = app.create_interaction_context()
            actual = await app.run_interaction("hello", context=context)

        self.assertEqual(actual, result)
        mock_orchestrate.assert_awaited_once()
        self.assertEqual(mock_orchestrate.await_args.args[1], context)
        self.assertEqual(mock_orchestrate.await_args.args[2], "hello")

        await app.__aexit__(None, None, None)

    async def test_run_interaction_emits_application_logs(self) -> None:
        result = InteractionResult(answer="Ready", tool_observations=[], used_step_limit_fallback=False)

        with (
            patch("application.service.McpService", _FakeMcpService),
            patch("application.service.build_system_prompt", return_value="system"),
            patch("application.service.orchestrate_interaction", new=AsyncMock(return_value=result)),
        ):
            app = WeekendWizardApp(Path("main.py"), "llama3.2:latest", ["mcp-server"])
            with self.assertLogs("weekend_wizard.application.service", level="INFO") as captured:
                await app.__aenter__()
                await app.run_interaction("hello", context=app.create_interaction_context())
                await app.__aexit__(None, None, None)

        joined = "\n".join(captured.output)
        self.assertIn("App session ready", joined)
        self.assertIn("Dispatching interaction", joined)
        self.assertIn("Interaction completed", joined)

    async def test_run_interaction_requires_explicit_context(self) -> None:
        with (
            patch("application.service.McpService", _FakeMcpService),
            patch("application.service.build_system_prompt", return_value="system"),
        ):
            app = WeekendWizardApp(Path("main.py"), "llama3.2:latest", ["mcp-server"])
            await app.__aenter__()

            with self.assertRaises(TypeError):
                await app.run_interaction("hello")  # type: ignore[call-arg]

        await app.__aexit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
