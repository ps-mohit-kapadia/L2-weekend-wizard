from __future__ import annotations

import unittest
from pathlib import Path

from api import build_readiness_response


class _FakeReadyApp:
    model_name = "llama3.2:latest"
    tool_names = ["get_weather", "book_recs"]
    server_path = Path("main.py")
    is_initialized = True


class HealthTests(unittest.TestCase):
    def test_build_readiness_response_reflects_ready_runtime(self) -> None:
        response = build_readiness_response(
            status="ready",
            server_path=Path("main.py"),
            model_name="llama3.2:latest",
            wizard=_FakeReadyApp(),
            details=None,
            provider_name="ollama",
            provider_reachable=True,
            model_available=True,
        )

        self.assertEqual(response.status, "ready")
        self.assertEqual(response.tool_count, 2)
        self.assertTrue(response.checks.model_resolved)
        self.assertTrue(response.checks.model_available)
        self.assertTrue(response.checks.server_path_exists)
        self.assertTrue(response.checks.ollama_reachable)
        self.assertTrue(response.checks.mcp_session_ready)
        self.assertTrue(response.checks.tools_discovered)
        self.assertIsNone(response.details)

    def test_build_readiness_response_captures_runtime_failure(self) -> None:
        response = build_readiness_response(
            status="not_ready",
            server_path=Path("main.py"),
            model_name="llama3.2:latest",
            wizard=None,
            details="startup boom",
            provider_name="ollama",
            provider_reachable=False,
            model_available=False,
        )

        self.assertEqual(response.status, "not_ready")
        self.assertTrue(response.checks.model_resolved)
        self.assertFalse(response.checks.model_available)
        self.assertTrue(response.checks.server_path_exists)
        self.assertFalse(response.checks.ollama_reachable)
        self.assertFalse(response.checks.mcp_session_ready)
        self.assertFalse(response.checks.tools_discovered)
        self.assertEqual(response.details, "startup boom")

    def test_build_readiness_response_treats_non_ollama_provider_as_not_applicable(self) -> None:
        response = build_readiness_response(
            status="ready",
            server_path=Path("main.py"),
            model_name="remote-model",
            wizard=None,
            details=None,
            provider_name="aiplatform",
            provider_reachable=False,
            model_available=False,
        )

        self.assertTrue(response.checks.model_available)
        self.assertTrue(response.checks.ollama_reachable)


if __name__ == "__main__":
    unittest.main()
