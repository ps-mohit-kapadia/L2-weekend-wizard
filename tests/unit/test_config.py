from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from config.config import get_settings


class ConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        get_settings.cache_clear()

    def test_get_settings_groups_environment_by_concern(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WEEKEND_WIZARD_HTTP_MAX_RETRIES": "4",
                "WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS": "0.25",
                "WEEKEND_WIZARD_MAX_STEPS": "7",
                "WEEKEND_WIZARD_LOG_LEVEL": "INFO",
                "OLLAMA_URL": "http://localhost:11434/api/chat",
            },
            clear=False,
        ):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.http.max_retries, 4)
        self.assertEqual(settings.http.retry_backoff_seconds, 0.25)
        self.assertEqual(settings.agent.max_steps, 7)
        self.assertEqual(settings.logging.level, "INFO")
        self.assertEqual(settings.llm.ollama_url, "http://localhost:11434/api/chat")

    def test_get_settings_reloads_when_environment_changes(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WEEKEND_WIZARD_HTTP_MAX_RETRIES": "6",
            },
            clear=False,
        ):
            get_settings.cache_clear()
            settings = get_settings()

            self.assertEqual(settings.http.max_retries, 6)


if __name__ == "__main__":
    unittest.main()
