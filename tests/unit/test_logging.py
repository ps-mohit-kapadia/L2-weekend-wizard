from __future__ import annotations

import logging
import sys
import unittest

from logger.logging import get_logger


class LoggingTests(unittest.TestCase):
    def test_get_logger_uses_project_namespace(self) -> None:
        logger = get_logger("tools.shared")

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "weekend_wizard.tools.shared")

    def test_get_logger_reuses_existing_handler(self) -> None:
        logger = get_logger("tools.shared")
        handler_count = len(logger.handlers)

        same_logger = get_logger("tools.shared")

        self.assertIs(logger, same_logger)
        self.assertEqual(len(same_logger.handlers), handler_count)

    def test_logger_uses_stderr_stream(self) -> None:
        logger = get_logger("tools.shared")

        self.assertEqual(len(logger.handlers), 1)
        self.assertIs(getattr(logger.handlers[0], "stream", None), sys.stderr)

    def test_logger_emits_plain_message(self) -> None:
        logger = get_logger("tools.shared")

        with self.assertLogs("weekend_wizard.tools.shared", level="INFO") as captured:
            logger.info("Starting HTTP request to %s", "https://example.com")

        joined = "\n".join(captured.output)
        self.assertIn("Starting HTTP request to https://example.com", joined)


if __name__ == "__main__":
    unittest.main()
