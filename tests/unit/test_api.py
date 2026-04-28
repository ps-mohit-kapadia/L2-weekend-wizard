from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
from schemas.agent import InteractionResult, ToolObservation
from schemas.api import ReadinessChecks, ReadinessResponse

_SETTINGS_WITH_KEY = type("Settings", (), {"api_key": "test-key"})()


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

    def create_interaction_context(self, request_id: str) -> object:
        context = object()
        self.created_contexts.append((request_id, context))
        return context


class _BrokenWizardApp(_FakeWizardApp):
    async def __aenter__(self) -> _BrokenWizardApp:
        raise RuntimeError("startup boom")


class ApiTests(unittest.TestCase):
    def test_health_endpoint_returns_ok(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.get_settings", return_value=_SETTINGS_WITH_KEY),
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
            patch("api.get_settings", return_value=_SETTINGS_WITH_KEY),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.list_available_models", return_value=["llama3.2:latest"]),
            TestClient(api.create_api()) as client,
        ):
            response = client.get("/ready", headers={"X-API-Key": "test-key"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.assertEqual(response.json()["tool_count"], 1)
        self.assertTrue(response.json()["checks"]["mcp_session_ready"])
        self.assertTrue(response.json()["checks"]["model_available"])

    def test_ready_endpoint_returns_503_when_not_ready(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.get_settings", return_value=_SETTINGS_WITH_KEY),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _BrokenWizardApp),
            TestClient(api.create_api()) as client,
        ):
            response = client.get("/ready", headers={"X-API-Key": "test-key"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "not_ready")
        self.assertIn("startup boom", response.json()["details"])

    def test_discover_model_failure_still_serves_health_and_not_ready(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.get_settings", return_value=_SETTINGS_WITH_KEY),
            patch("api.discover_model", side_effect=RuntimeError("ollama offline")),
            TestClient(api.create_api()) as client,
        ):
            health_response = client.get("/health")
            ready_response = client.get("/ready", headers={"X-API-Key": "test-key"})

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(ready_response.status_code, 503)
        self.assertEqual(ready_response.json()["status"], "not_ready")
        self.assertIn("ollama offline", ready_response.json()["details"])

    def test_chat_endpoint_returns_structured_response(self) -> None:
        fake_app = _FakeWizardApp()

        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.get_settings", return_value=_SETTINGS_WITH_KEY),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", return_value=fake_app),
            patch("api.list_available_models", return_value=["llama3.2:latest"]),
            TestClient(api.create_api()) as client,
        ):
            with self.assertLogs("weekend_wizard.agent.api", level="INFO") as captured:
                response = client.post(
                    "/chat",
                    json={"prompt": "Plan me a weekend in New York"},
                    headers={"X-API-Key": "test-key"},
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Weekend plan ready.")
        self.assertEqual(response.json()["tool_observations"][0]["tool_name"], "get_weather")
        self.assertEqual(response.json()["response_status"], "success")
        self.assertEqual(len(fake_app.created_contexts), 1)
        request_id, created_context = fake_app.created_contexts[0]
        self.assertTrue(request_id)
        fake_app.run_interaction.assert_awaited_once_with("Plan me a weekend in New York", context=created_context)
        joined = "\n".join(captured.output)
        self.assertIn("Received /chat request", joined)
        self.assertIn("Completed /chat request", joined)

    def test_chat_endpoint_exposes_fallback_state(self) -> None:
        fake_app = _FakeWizardApp()
        fake_app.run_interaction = AsyncMock(
            return_value=InteractionResult(
                answer="I could not complete the full plan, but here is a safe fallback.",
                tool_observations=[],
                used_fallback=True,
            )
        )

        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.get_settings", return_value=_SETTINGS_WITH_KEY),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", return_value=fake_app),
            patch("api.list_available_models", return_value=["llama3.2:latest"]),
            TestClient(api.create_api()) as client,
        ):
            response = client.post(
                "/chat",
                json={"prompt": "Plan a weekend in Las Vegas with 3 adventure books."},
                headers={"X-API-Key": "test-key"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response_status"], "degraded")

    def test_chat_endpoint_surfaces_server_errors(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.get_settings", return_value=_SETTINGS_WITH_KEY),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _BrokenWizardApp),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"}, headers={"X-API-Key": "test-key"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "startup boom")

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
            patch("api.get_settings", return_value=_SETTINGS_WITH_KEY),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", return_value=fake_app),
            patch("api.evaluate_runtime_readiness", side_effect=[ready_response]),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"}, headers={"X-API-Key": "test-key"})

        self.assertEqual(response.status_code, 200)

    def test_protected_endpoints_reject_missing_api_key(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.get_settings", return_value=_SETTINGS_WITH_KEY),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.list_available_models", return_value=["llama3.2:latest"]),
            TestClient(api.create_api()) as client,
        ):
            ready_response = client.get("/ready")
            chat_response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(ready_response.status_code, 401)
        self.assertEqual(chat_response.status_code, 401)

    def test_protected_endpoints_fail_when_server_api_key_is_not_configured(self) -> None:
        settings_without_key = type("Settings", (), {"api_key": ""})()
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.get_settings", return_value=settings_without_key),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.list_available_models", return_value=["llama3.2:latest"]),
            TestClient(api.create_api()) as client,
        ):
            response = client.get("/ready", headers={"X-API-Key": "test-key"})

        self.assertEqual(response.status_code, 503)
        self.assertIn("API key is not configured", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
