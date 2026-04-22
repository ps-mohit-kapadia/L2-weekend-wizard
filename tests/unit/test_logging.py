from __future__ import annotations

import logging
import os
import sys
import unittest
from unittest.mock import patch

from logger import logging as logging_module


class LoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        logging_module.get_settings.cache_clear()
        logging_module._REQUEST_ID.set(None)

    def tearDown(self) -> None:
        logging_module.get_settings.cache_clear()
        logging_module._REQUEST_ID.set(None)

    def test_get_logger_uses_project_namespace(self) -> None:
        logger = logging_module.get_logger("tools.shared")

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "weekend_wizard.tools.shared")

    def test_get_logger_reuses_existing_handler(self) -> None:
        logger = logging_module.get_logger("tools.shared")
        handler_count = len(logger.handlers)

        same_logger = logging_module.get_logger("tools.shared")

        self.assertIs(logger, same_logger)
        self.assertEqual(len(same_logger.handlers), handler_count)

    def test_logger_uses_stderr_stream(self) -> None:
        logger = logging_module.get_logger("tools.shared")

        self.assertEqual(len(logger.handlers), 1)
        self.assertIs(getattr(logger.handlers[0], "stream", None), sys.stderr)

    def test_local_mode_keeps_readable_message(self) -> None:
        logger = logging_module.get_logger("tools.shared")

        with self.assertLogs("weekend_wizard.tools.shared", level="INFO") as captured:
            logger.info("Starting HTTP request to %s", "https://example.com")

        joined = "\n".join(captured.output)
        self.assertIn("Starting HTTP request to https://example.com", joined)

    def test_staging_mode_formatter_appends_request_context_and_fields(self) -> None:
        with patch.dict(os.environ, {"WEEKEND_WIZARD_OBSERVABILITY_MODE": "staging"}, clear=False):
            logging_module.get_settings.cache_clear()
            logger = logging_module.get_logger("logger.observability")
            token = logging_module.set_request_context("req-123")
            try:
                record = logger.makeRecord(
                    logger.name,
                    logging.INFO,
                    __file__,
                    42,
                    "Planner completed",
                    args=(),
                    exc_info=None,
                    extra=logging_module.get_log_extra(
                        event="planner_completed",
                        phase="planner",
                        outcome="success",
                        duration_ms=125,
                    ),
                )
                logger.handlers[0].filter(record)
                formatted = logger.handlers[0].format(record)
            finally:
                logging_module.reset_request_context(token)

        self.assertIn("Planner completed", formatted)
        self.assertIn("request_id=req-123", formatted)
        self.assertIn("event=planner_completed", formatted)
        self.assertIn("phase=planner", formatted)
        self.assertIn("outcome=success", formatted)
        self.assertIn("duration_ms=125", formatted)

    def test_mode_helpers_distinguish_staging_and_production(self) -> None:
        with patch.dict(os.environ, {"WEEKEND_WIZARD_OBSERVABILITY_MODE": "staging"}, clear=False):
            logging_module.get_settings.cache_clear()
            self.assertTrue(logging_module.staging_mode())
            self.assertFalse(logging_module.production_mode())

        with patch.dict(os.environ, {"WEEKEND_WIZARD_OBSERVABILITY_MODE": "production"}, clear=False):
            logging_module.get_settings.cache_clear()
            self.assertFalse(logging_module.staging_mode())
            self.assertTrue(logging_module.production_mode())


if __name__ == "__main__":
    unittest.main()
