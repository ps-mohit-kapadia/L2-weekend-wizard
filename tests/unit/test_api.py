from __future__ import annotations

import asyncio
import time
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


class _SlowWizardApp(_FakeWizardApp):
    async def __aenter__(self) -> _SlowWizardApp:
        await asyncio.sleep(0.2)
        self.is_initialized = True
        return self


class _ExplodingWizardApp(_FakeWizardApp):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__(*_args, **_kwargs)
        self.run_interaction = AsyncMock(side_effect=RuntimeError("internal boom"))


class ApiTests(unittest.TestCase):
    def _wait_for_ready_status(
        self,
        client: TestClient,
        expected_status: str,
        *,
        timeout_seconds: float = 1.0,
    ) -> dict:
        deadline = time.monotonic() + timeout_seconds
        last_payload: dict = {}
        while time.monotonic() < deadline:
            response = client.get("/ready")
            last_payload = response.json()
            if last_payload.get("status") == expected_status:
                return last_payload
            time.sleep(0.02)
        self.fail(f"Timed out waiting for /ready status {expected_status!r}. Last payload: {last_payload}")

    def test_ready_endpoint_returns_503_when_model_discovery_fails_during_startup(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", side_effect=RuntimeError("offline")),
            TestClient(api.create_api()) as client,
        ):
            payload = self._wait_for_ready_status(client, "not_ready")

        self.assertEqual(payload["status"], "not_ready")
        self.assertEqual(payload["details"], api.UNEXPECTED_READINESS_ERROR_DETAIL)

    def test_health_endpoint_returns_ok(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            TestClient(api.create_api()) as client,
        ):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_health_is_reachable_while_runtime_is_still_warming(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _SlowWizardApp),
            TestClient(api.create_api()) as client,
        ):
            health_response = client.get("/health")
            ready_response = client.get("/ready")

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(health_response.json(), {"status": "ok"})
        self.assertEqual(ready_response.status_code, 503)
        self.assertEqual(ready_response.json()["status"], "not_ready")
        self.assertEqual(ready_response.json()["details"], api.STARTING_READINESS_DETAIL)

    def test_ready_endpoint_returns_structured_readiness_when_ready(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            TestClient(api.create_api()) as client,
        ):
            payload = self._wait_for_ready_status(client, "ready")

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["tool_count"], 1)
        self.assertTrue(payload["checks"]["mcp_session_ready"])
        self.assertTrue(payload["checks"]["model_available"])
        self.assertTrue(payload["checks"]["ollama_reachable"])

    def test_ready_endpoint_returns_503_when_not_ready(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _BrokenWizardApp),
            TestClient(api.create_api()) as client,
        ):
            payload = self._wait_for_ready_status(client, "not_ready")

        self.assertEqual(payload["status"], "not_ready")
        self.assertEqual(payload["details"], api.UNEXPECTED_READINESS_ERROR_DETAIL)

    def test_chat_endpoint_returns_structured_response(self) -> None:
        fake_app = _FakeWizardApp()

        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", return_value=fake_app),
            TestClient(api.create_api()) as client,
        ):
            with self.assertLogs("weekend_wizard.agent.api", level="INFO") as captured:
                response = client.post("/chat", json={"prompt": "Plan me a weekend in New York"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "Weekend plan ready.")
        self.assertEqual(response.json()["tool_observations"][0]["tool_name"], "get_weather")
        self.assertEqual(len(fake_app.created_contexts), 1)
        created_context = fake_app.created_contexts[0]
        fake_app.run_interaction.assert_awaited_once()
        self.assertEqual(
            fake_app.run_interaction.await_args.args[0],
            "Plan me a weekend in New York",
        )
        self.assertEqual(
            fake_app.run_interaction.await_args.kwargs["context"],
            created_context,
        )
        self.assertIn("trace", fake_app.run_interaction.await_args.kwargs)
        joined = "\n".join(captured.output)
        self.assertIn("Received /chat request", joined)
        self.assertIn("Completed /chat request", joined)
        self.assertIn("REQUEST TRACE:", joined)
        self.assertIn("EVENT: interaction_started", joined)
        self.assertIn("EVENT: interaction_completed", joined)

    def test_chat_endpoint_emits_trace_even_when_interaction_fails(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _ExplodingWizardApp),
            TestClient(api.create_api()) as client,
        ):
            with self.assertLogs("weekend_wizard.agent.api", level="INFO") as captured:
                response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 500)
        joined = "\n".join(captured.output)
        self.assertIn("REQUEST TRACE:", joined)
        self.assertIn("EVENT: interaction_started", joined)
        self.assertIn("EVENT: interaction_completed", joined)

    def test_chat_endpoint_surfaces_server_errors(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _SlowWizardApp),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], api.STARTING_READINESS_DETAIL)

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
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], api.UNEXPECTED_CHAT_ERROR_DETAIL)

    def test_ready_endpoint_does_not_recompute_readiness_after_startup(self) -> None:
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
            TestClient(api.create_api()) as client,
        ):
            app_instance = client.app
            app_instance.state.readiness = ready_response
            response = client.get("/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), ready_response.model_dump())

    def test_ready_endpoint_reports_ready_for_aiplatform_without_ollama_probe(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.get_settings") as mock_get_settings,
            patch("api.discover_model", return_value="gpt-like-model"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            TestClient(api.create_api()) as client,
        ):
            mock_get_settings.return_value.llm_provider = "aiplatform"
            payload = self._wait_for_ready_status(client, "ready")

        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["checks"]["model_available"])
        self.assertTrue(payload["checks"]["ollama_reachable"])


if __name__ == "__main__":
    unittest.main()
