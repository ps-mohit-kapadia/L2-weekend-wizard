from __future__ import annotations

import asyncio
from dataclasses import replace
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
from config.config import get_settings
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


class _NeverCompletingWizardApp(_FakeWizardApp):
    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__(*_args, **_kwargs)
        self.run_interaction = AsyncMock(side_effect=self._never_complete)

    async def _never_complete(self, *_args, **_kwargs) -> None:
        await asyncio.sleep(10)


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._settings_patch = patch(
            "api.get_settings",
            return_value=replace(get_settings(), api_key=None, llm_provider="ollama"),
        )
        self._settings_patch.start()

    def tearDown(self) -> None:
        self._settings_patch.stop()

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
            patch("api.discover_model", side_effect=RuntimeError("offline")) as mock_discover_model,
            TestClient(api.create_api()) as client,
        ):
            payload = self._wait_for_ready_status(client, "not_ready")

        self.assertEqual(payload["status"], "not_ready")
        self.assertEqual(payload["details"], api.UNEXPECTED_READINESS_ERROR_DETAIL)
        self.assertEqual(mock_discover_model.call_count, 1)

    def test_ready_endpoint_recovers_when_startup_failure_is_retryable(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.STARTUP_RETRY_INTERVAL_SECONDS", 0),
            patch(
                "api.discover_model",
                side_effect=[
                    RuntimeError("Could not reach Ollama to validate model 'llama3.2:latest': offline"),
                    "llama3.2:latest",
                ],
            ) as mock_discover_model,
            patch("api.WeekendWizardApp", _FakeWizardApp),
            TestClient(api.create_api()) as client,
        ):
            payload = self._wait_for_ready_status(client, "ready")

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(mock_discover_model.call_count, 2)

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
        self.assertEqual(payload["provider"], "ollama")
        self.assertEqual(payload["tool_count"], 1)
        self.assertTrue(payload["checks"]["mcp_session_ready"])
        self.assertTrue(payload["checks"]["model_available"])
        self.assertTrue(payload["checks"]["provider_reachable"])
        self.assertTrue(payload["checks"]["ollama_reachable"])
        self.assertTrue(payload["checks"]["rate_limit_configured"])
        self.assertTrue(payload["checks"]["trace_logging_configured"])

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
        self.assertTrue(response.json()["correlation_id"].startswith("corr_"))
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
        self.assertIn("correlation_id=corr_", joined)
        self.assertIn("interface=chat", joined)
        self.assertIn("CORRELATION ID:", joined)
        self.assertIn("EVENT: interaction_started", joined)
        self.assertIn("EVENT: interaction_completed", joined)

    def test_chat_endpoint_allows_requests_without_configured_api_key(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 200)

    def test_chat_endpoint_rejects_missing_api_key_when_configured(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.get_settings", return_value=replace(get_settings(), api_key="secret")),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], api.UNAUTHORIZED_DETAIL)

    def test_chat_endpoint_rejects_wrong_api_key_when_configured(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.get_settings", return_value=replace(get_settings(), api_key="secret")),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"}, headers={"X-API-Key": "wrong"})

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], api.UNAUTHORIZED_DETAIL)

    def test_chat_endpoint_accepts_configured_api_key(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.get_settings", return_value=replace(get_settings(), api_key="secret")),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"}, headers={"X-API-Key": "secret"})

        self.assertEqual(response.status_code, 200)

    def test_health_and_ready_do_not_require_api_key(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.get_settings", return_value=replace(get_settings(), api_key="secret")),
            TestClient(api.create_api()) as client,
        ):
            health_response = client.get("/health")
            ready_payload = self._wait_for_ready_status(client, "ready")

        self.assertEqual(health_response.status_code, 200)
        self.assertEqual(ready_payload["status"], "ready")

    def test_chat_endpoint_rejects_prompt_over_configured_limit(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.get_settings", return_value=replace(get_settings(), api_key=None, max_prompt_chars=5)),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "too long"})

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], api.PROMPT_TOO_LARGE_DETAIL)

    def test_chat_endpoint_rate_limits_by_client(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.get_settings", return_value=replace(get_settings(), api_key=None, rate_limit_requests=1)),
            TestClient(api.create_api()) as client,
        ):
            first_response = client.post("/chat", json={"prompt": "hello"})
            second_response = client.post("/chat", json={"prompt": "hello again"})

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 429)
        self.assertEqual(second_response.json()["detail"], api.RATE_LIMITED_DETAIL)

    def test_chat_endpoint_returns_504_when_interaction_times_out(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _NeverCompletingWizardApp),
            patch("api.get_settings", return_value=replace(get_settings(), api_key=None, request_timeout=0.001)),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/chat", json={"prompt": "hello"})

        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["detail"], api.REQUEST_TIMEOUT_DETAIL)

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
        self.assertIn("CORRELATION ID:", joined)
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
            provider="ollama",
            model_name="llama3.2:latest",
            tool_count=1,
            request_timeout_seconds=1200,
            rate_limit_requests=20,
            rate_limit_window_seconds=60,
            checks=ReadinessChecks(
                model_resolved=True,
                model_available=True,
                server_path_exists=True,
                provider_reachable=True,
                ollama_reachable=True,
                mcp_session_ready=True,
                tools_discovered=True,
                auth_configured=False,
                rate_limit_configured=True,
                trace_logging_configured=True,
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
            patch("api.get_settings", return_value=replace(get_settings(), llm_provider="aiplatform")),
            patch("api.discover_model", return_value="gpt-like-model"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            TestClient(api.create_api()) as client,
        ):
            payload = self._wait_for_ready_status(client, "ready")

        self.assertEqual(payload["status"], "ready")
        self.assertEqual(payload["provider"], "aiplatform")
        self.assertTrue(payload["checks"]["model_available"])
        self.assertTrue(payload["checks"]["provider_reachable"])
        self.assertTrue(payload["checks"]["ollama_reachable"])


if __name__ == "__main__":
    unittest.main()
