from __future__ import annotations

import unittest

from logger.context import request_context
from logger.logging import ProjectLogger, format_event, get_logger


class LoggingTests(unittest.TestCase):
    def test_get_logger_uses_project_namespace(self) -> None:
        logger = get_logger("tools.shared", layer="tools")

        self.assertIsInstance(logger, ProjectLogger)
        self.assertEqual(logger.name, "weekend_wizard.tools.shared")

    def test_format_event_serializes_fields_consistently(self) -> None:
        message = format_event(
            "sample_event",
            answer_length=12,
            used_fallback=False,
            tool_names=["get_weather", "book_recs"],
            details="ok",
        )

        self.assertIn("event=sample_event", message)
        self.assertIn("answer_length=12", message)
        self.assertIn("used_fallback=false", message)
        self.assertIn('tool_names=["get_weather", "book_recs"]', message)
        self.assertIn('details="ok"', message)

    def test_project_logger_includes_layer_and_request_context(self) -> None:
        logger = get_logger("tools.shared", layer="tools")

        with request_context("test"):
            with self.assertLogs("weekend_wizard.tools.shared", level="INFO") as captured:
                logger.info("request_start", url="https://example.com")

        joined = "\n".join(captured.output)
        self.assertIn("event=request_start", joined)
        self.assertIn('layer="tools"', joined)
        self.assertRegex(joined, r'request_id="test-[0-9a-f]{8}"')


if __name__ == "__main__":
    unittest.main()
