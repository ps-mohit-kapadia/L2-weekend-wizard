from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from api import build_not_ready_response, evaluate_runtime_readiness


class _FakeReadyApp:
    model_name = "llama3.2:latest"
    tool_names = ["get_weather", "book_recs"]
    server_path = Path("main.py")
    is_initialized = True


class HealthTests(unittest.TestCase):
    def test_evaluate_runtime_readiness_reflects_initialized_runtime(self) -> None:
        with patch("api.list_available_models", return_value=["llama3.2:latest"]):
            response = evaluate_runtime_readiness(_FakeReadyApp())

        self.assertEqual(response.status, "ready")
        self.assertEqual(response.tool_count, 2)
        self.assertTrue(response.checks.model_resolved)
        self.assertTrue(response.checks.model_available)
        self.assertTrue(response.checks.server_path_exists)
        self.assertTrue(response.checks.ollama_reachable)
        self.assertTrue(response.checks.mcp_session_ready)
        self.assertTrue(response.checks.tools_discovered)
        self.assertIsNone(response.details)

    def test_build_not_ready_response_captures_runtime_failure(self) -> None:
        response = build_not_ready_response(Path("main.py"), "llama3.2:latest", "startup boom")

        self.assertEqual(response.status, "not_ready")
        self.assertTrue(response.checks.model_resolved)
        self.assertFalse(response.checks.model_available)
        self.assertTrue(response.checks.server_path_exists)
        self.assertFalse(response.checks.ollama_reachable)
        self.assertFalse(response.checks.mcp_session_ready)
        self.assertFalse(response.checks.tools_discovered)
        self.assertEqual(response.details, "startup boom")

    def test_evaluate_runtime_readiness_returns_not_ready_when_model_is_missing(self) -> None:
        with patch("api.list_available_models", return_value=["different-model"]):
            response = evaluate_runtime_readiness(_FakeReadyApp())

        self.assertEqual(response.status, "not_ready")
        self.assertTrue(response.checks.ollama_reachable)
        self.assertFalse(response.checks.model_available)
        self.assertIn("Resolved model is not available", response.details or "")

    def test_evaluate_runtime_readiness_returns_not_ready_when_ollama_is_unreachable(self) -> None:
        with patch("api.list_available_models", side_effect=requests.RequestException("offline")):
            response = evaluate_runtime_readiness(_FakeReadyApp())

        self.assertEqual(response.status, "not_ready")
        self.assertFalse(response.checks.ollama_reachable)
        self.assertFalse(response.checks.model_available)
        self.assertIn("Ollama is not reachable", response.details or "")


if __name__ == "__main__":
    unittest.main()
