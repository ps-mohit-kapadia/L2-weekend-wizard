from __future__ import annotations

import unittest

from schemas.api import ChatRequest, ChatResponse, HealthResponse, ReadinessChecks, ReadinessResponse


class ApiSchemaTests(unittest.TestCase):
    def test_chat_request_requires_non_empty_prompt(self) -> None:
        request = ChatRequest(prompt="hello")

        self.assertEqual(request.prompt, "hello")

    def test_chat_response_captures_tool_observations(self) -> None:
        response = ChatResponse(correlation_id="corr_1234567890abcdef", answer="done")

        self.assertEqual(response.correlation_id, "corr_1234567890abcdef")
        self.assertEqual(response.answer, "done")
        self.assertEqual(response.tool_observations, [])

    def test_health_response_defaults_are_explicit(self) -> None:
        response = HealthResponse(status="ok")

        self.assertEqual(response.status, "ok")

    def test_readiness_response_captures_detailed_checks(self) -> None:
        response = ReadinessResponse(
            status="ready",
            provider="ollama",
            model_name="llama3.2:latest",
            tool_count=6,
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
        )

        self.assertEqual(response.status, "ready")
        self.assertEqual(response.provider, "ollama")
        self.assertEqual(response.model_name, "llama3.2:latest")
        self.assertEqual(response.tool_count, 6)
        self.assertEqual(response.request_timeout_seconds, 1200)
        self.assertTrue(response.checks.model_available)
        self.assertTrue(response.checks.provider_reachable)
        self.assertTrue(response.checks.tools_discovered)


if __name__ == "__main__":
    unittest.main()
