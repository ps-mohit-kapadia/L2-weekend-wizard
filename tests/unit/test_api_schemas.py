from __future__ import annotations

import unittest

from schemas.api import ChatRequest, ChatResponse, HealthResponse, ReadinessChecks, ReadinessResponse


class ApiSchemaTests(unittest.TestCase):
    def test_chat_request_requires_non_empty_prompt(self) -> None:
        request = ChatRequest(prompt="hello")

        self.assertEqual(request.prompt, "hello")

    def test_chat_response_captures_tool_observations(self) -> None:
        response = ChatResponse(answer="done")

        self.assertEqual(response.answer, "done")
        self.assertEqual(response.tool_observations, [])
        self.assertFalse(response.used_step_limit_fallback)

    def test_health_response_defaults_are_explicit(self) -> None:
        response = HealthResponse(status="ok")

        self.assertEqual(response.status, "ok")

    def test_readiness_response_captures_detailed_checks(self) -> None:
        response = ReadinessResponse(
            status="ready",
            model_name="llama3.2:latest",
            tool_count=6,
            checks=ReadinessChecks(
                model_resolved=True,
                model_available=True,
                server_path_exists=True,
                ollama_reachable=True,
                mcp_session_ready=True,
                tools_discovered=True,
            ),
        )

        self.assertEqual(response.status, "ready")
        self.assertEqual(response.model_name, "llama3.2:latest")
        self.assertEqual(response.tool_count, 6)
        self.assertTrue(response.checks.model_available)
        self.assertTrue(response.checks.tools_discovered)


if __name__ == "__main__":
    unittest.main()
