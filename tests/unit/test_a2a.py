from __future__ import annotations

import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import api
from config.config import get_settings
from schemas.agent import InteractionResult


class _FakeWizardApp:
    def __init__(self, *_args, **_kwargs) -> None:
        self.model_name = "llama3.2:latest"
        self.tool_names = ["trivia"]
        self.server_path = Path("main.py")
        self.is_initialized = False
        self.run_interaction = AsyncMock(
            return_value=InteractionResult(answer="Trivia: demo. Answer: demo.")
        )

    async def __aenter__(self) -> _FakeWizardApp:
        self.is_initialized = True
        return self

    async def __aexit__(self, exc_type, exc, exc_tb) -> None:
        self.is_initialized = False

    def create_interaction_context(self) -> object:
        return object()


class A2ATests(unittest.TestCase):
    def setUp(self) -> None:
        self._settings_patch = patch(
            "api.get_settings",
            return_value=replace(get_settings(), api_key=None, llm_provider="ollama"),
        )
        self._settings_patch.start()

    def tearDown(self) -> None:
        self._settings_patch.stop()

    def _wait_for_ready_status(self, client: TestClient, expected_status: str) -> dict:
        deadline = time.monotonic() + 1.0
        last_payload: dict = {}
        while time.monotonic() < deadline:
            response = client.get("/ready")
            last_payload = response.json()
            if last_payload.get("status") == expected_status:
                return last_payload
            time.sleep(0.02)
        self.fail(f"Timed out waiting for /ready status {expected_status!r}. Last payload: {last_payload}")

    def _message_send_payload(self, prompt: str = "Give me one trivia question.") -> dict:
        return {
            "jsonrpc": "2.0",
            "id": "req-1",
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": prompt}],
                }
            },
        }

    def test_agent_card_returns_weekend_wizard_capabilities(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            TestClient(api.create_api()) as client,
        ):
            response = client.get("/.well-known/agent.json")

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["name"], "Weekend Wizard")
        self.assertEqual(payload["url"], "http://testserver/a2a/jsonrpc")
        self.assertFalse(payload["capabilities"]["streaming"])
        self.assertIn("text/plain", payload["defaultInputModes"])
        self.assertIn("weekend_planning", {skill["id"] for skill in payload["skills"]})

    def test_a2a_message_send_returns_completed_artifact(self) -> None:
        fake_app = _FakeWizardApp()
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", return_value=fake_app),
            TestClient(api.create_api()) as client,
        ):
            self._wait_for_ready_status(client, "ready")
            with self.assertLogs("weekend_wizard.agent.api", level="INFO") as captured:
                response = client.post("/a2a/jsonrpc", json=self._message_send_payload())

        payload = response.json()
        joined = "\n".join(captured.output)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["jsonrpc"], "2.0")
        self.assertEqual(payload["id"], "req-1")
        self.assertEqual(payload["result"]["status"]["state"], "completed")
        self.assertTrue(payload["result"]["correlation_id"].startswith("corr_"))
        self.assertEqual(
            payload["result"]["artifacts"][0]["parts"][0]["text"],
            "Trivia: demo. Answer: demo.",
        )
        self.assertIn("correlation_id=corr_", joined)
        self.assertIn("interface=a2a", joined)
        fake_app.run_interaction.assert_awaited_once()

    def test_a2a_rejects_invalid_jsonrpc_version(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            TestClient(api.create_api()) as client,
        ):
            payload = self._message_send_payload()
            payload["jsonrpc"] = "1.0"
            response = client.post("/a2a/jsonrpc", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"]["code"], -32600)

    def test_a2a_rejects_unsupported_method(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            TestClient(api.create_api()) as client,
        ):
            payload = self._message_send_payload()
            payload["method"] = "tasks/get"
            response = client.post("/a2a/jsonrpc", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"]["code"], -32601)

    def test_a2a_rejects_missing_text_part(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            TestClient(api.create_api()) as client,
        ):
            payload = self._message_send_payload()
            payload["params"]["message"]["parts"] = []
            response = client.post("/a2a/jsonrpc", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["error"]["code"], -32602)

    def test_a2a_requires_api_key_when_configured(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.get_settings", return_value=replace(get_settings(), api_key="secret")),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/a2a/jsonrpc", json=self._message_send_payload())

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], api.UNAUTHORIZED_DETAIL)

    def test_a2a_prompt_limit_uses_shared_boundary_control(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.get_settings", return_value=replace(get_settings(), api_key=None, max_prompt_chars=5)),
            TestClient(api.create_api()) as client,
        ):
            response = client.post("/a2a/jsonrpc", json=self._message_send_payload("too long"))

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], api.PROMPT_TOO_LARGE_DETAIL)

    def test_a2a_rate_limit_uses_shared_boundary_control(self) -> None:
        with (
            patch("api.Path.resolve", return_value=Path("C:/project/api.py")),
            patch("api.discover_model", return_value="llama3.2:latest"),
            patch("api.WeekendWizardApp", _FakeWizardApp),
            patch("api.get_settings", return_value=replace(get_settings(), api_key=None, rate_limit_requests=1)),
            TestClient(api.create_api()) as client,
        ):
            self._wait_for_ready_status(client, "ready")
            first_response = client.post("/a2a/jsonrpc", json=self._message_send_payload())
            second_response = client.post("/a2a/jsonrpc", json=self._message_send_payload("Tell me one joke."))

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 429)
        self.assertEqual(second_response.json()["detail"], api.RATE_LIMITED_DETAIL)


if __name__ == "__main__":
    unittest.main()
