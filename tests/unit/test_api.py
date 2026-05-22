from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
from schemas.agent import InteractionResult, ToolObservation
from schemas.api import ReadinessChecks, ReadinessResponse


class _FakeWizardApp:
    def __init__(self, *_args, **_kwargs) -> None:
        self.model_name = "llama3.2:latest"
        self.tool_names = ["get_weather"]
        self.server_path = Path("main.py")
        self.is_initialized = False
        self.run_interaction = AsyncMock(
            return_value=InteractionResult(
                answer="Weekend plan ready.",
                tool_observations=[
                    ToolObservation(
                        tool_name="get_weather",
                        args={"latitude": 40.7, "longitude": -74.0},
                        payload='{"summary": "clear sky"}',
                    )
                ],
            )
        )
        self.created_contexts = []

    async def __aenter__(self) -> _FakeWizardApp:
        self.is_initialized = True
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        self.is_initialized = False
        return None

    def create_interaction_context(self) -> object:
        context = object()
        self.created_contexts.append(context)
        return context


class _BrokenWizardApp(_FakeWizardApp):
    async def __aenter__(self) -> _BrokenWizardApp:
        raise RuntimeError("startup boom")


class _ExplodingWizardApp(_FakeWizardApp):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__(*_args, **_kwargs)
        self.run_interaction = AsyncMock(side_effect=RuntimeError("internal boom"))


class ApiTests(unittest.TestCase):
    def test_ready_endpoint_returns_503_when_model_discovery_fails_during_startup(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", side_effect=RuntimeError("offline")),
            TestClient(api.create_api()) as client,
        ):
            response = client.get("/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "not_ready")
        self.assertEqual(response.json()["details"], api.UNEXPECTED_READINESS_ERROR_DETAIL)

    def test_health_endpoint_returns_ok(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.list_available_models", return_value=["llama3.2:latest"]),
            TestClient(api.create_api()) as client,
        ):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ready_endpoint_returns_structured_readiness_when_ready(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.list_available_models", return_value=["llama3.2:latest"]) as mock_list_models,
            TestClient(api.create_api()) as client,
        ):
            response = client.get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.assertEqual(response.json()["tool_count"], 1)
        self.assertTrue(response.json()["checks"]["mcp_session_ready"])
        self.assertTrue(response.json()["checks"]["model_available"])
        self.assertTrue(response.json()["checks"]["ollama_reachable"])
        mock_list_models.assert_called_once_with(timeout=5)

    def test_ready_endpoint_returns_503_when_not_ready(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _BrokenWizardApp),
            TestClient(api.create_api()) as client,
        ):
            response = client.get("/ready")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "not_ready")
        self.assertEqual(response.json()["details"], api.UNEXPECTED_READINESS_ERROR_DETAIL)

    def test_chat_endpoint_returns_structured_response(self) -> None:
        fake_app = _FakeWizardApp()

        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", return_value=fake_app),
            patch("api.list_available_models") as mock_list_models,
            TestClient(api.create_api()) as client,
        ):
            with self.assertLogs("weekend_wizard.agent.api", level="INFO") as captured:
                response = client.post("/chat", json={"prompt": "Plan me a weekend in New York"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Weekend plan ready.")
        self.assertEqual(response.json()["tool_observations"][0]["tool_name"], "get_weather")
        self.assertEqual(len(fake_app.created_contexts), 1)
        created_context = fake_app.created_contexts[0]
        fake_app.run_interaction.assert_awaited_once_with("Plan me a weekend in New York", context=created_context)
        mock_list_models.assert_not_called()
        joined = "\n".join(captured.output)
        self.assertIn("Received /chat request", joined)
        self.assertIn("Completed /chat request", joined)

    def test_chat_endpoint_surfaces_server_errors(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _BrokenWizardApp),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], api.UNEXPECTED_READINESS_ERROR_DETAIL)

    def test_chat_endpoint_returns_sanitized_not_ready_detail_after_startup_discovery_failure(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", side_effect=RuntimeError("offline")),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], api.UNEXPECTED_READINESS_ERROR_DETAIL)

    def test_chat_endpoint_sanitizes_unexpected_internal_errors(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _ExplodingWizardApp),
            patch("api.list_available_models") as mock_list_models,
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], api.UNEXPECTED_CHAT_ERROR_DETAIL)
        mock_list_models.assert_not_called()

    def test_chat_endpoint_does_not_recompute_full_readiness_per_request(self) -> None:
        fake_app = _FakeWizardApp()
        ready_response = ReadinessResponse(
            status="ready",
            model_name="llama3.2:latest",
            tool_count=1,
            checks=ReadinessChecks(
                model_resolved=True,
                model_available=True,
                server_path_exists=True,
                ollama_reachable=True,
                mcp_session_ready=True,
                tools_discovered=True,
            ),
            details=None,
        )

        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", return_value=fake_app),
            patch("api.evaluate_runtime_readiness", side_effect=[ready_response]),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
