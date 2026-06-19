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
                "WEEKEND_WIZARD_REQUEST_TIMEOUT": "900",
                "WEEKEND_WIZARD_TOOL_HTTP_TIMEOUT": "15",
                "WEEKEND_WIZARD_HTTP_MAX_RETRIES": "4",
                "WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS": "0.25",
                "WEEKEND_WIZARD_LOG_LEVEL": "INFO",
                "OLLAMA_URL": "http://localhost:11434/api/chat",
            },
            clear=True,
        ):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.request_timeout, 900)
        self.assertEqual(settings.tool_http_timeout, 15)
        self.assertEqual(settings.http_max_retries, 4)
        self.assertEqual(settings.http_retry_backoff_seconds, 0.25)
        self.assertEqual(settings.log_level, "INFO")
        self.assertIsNone(settings.api_key)
        self.assertEqual(settings.max_prompt_chars, 4000)
        self.assertEqual(settings.rate_limit_requests, 20)
        self.assertEqual(settings.rate_limit_window_seconds, 60)
        self.assertEqual(settings.llm_provider, "ollama")
        self.assertEqual(settings.ollama_url, "http://localhost:11434/api/chat")
        self.assertEqual(settings.aiplatform_base_url, "https://aiapidev.3ecompany.com")
        self.assertEqual(settings.aiplatform_timeout, 120)
        self.assertEqual(settings.aiplatform_chat_path, "/v1/chat/completions")
        self.assertEqual(settings.preferred_models, ("llama3.1:8b",))

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

            self.assertEqual(settings.http_max_retries, 6)

    def test_get_settings_uses_timeout_defaults_for_separate_domains(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.request_timeout, 1200)
        self.assertEqual(settings.tool_http_timeout, 20)
        self.assertEqual(settings.http_max_retries, 2)
        self.assertEqual(settings.http_retry_backoff_seconds, 0.5)
        self.assertEqual(settings.log_level, "WARNING")
        self.assertIsNone(settings.api_key)
        self.assertEqual(settings.max_prompt_chars, 4000)
        self.assertEqual(settings.rate_limit_requests, 20)
        self.assertEqual(settings.rate_limit_window_seconds, 60)
        self.assertEqual(settings.llm_provider, "ollama")
        self.assertEqual(settings.preferred_models, ("llama3.1:8b",))

    def test_get_settings_supports_aiplatform_provider_configuration(self) -> None:
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "aiplatform",
                "AIPLATFORM_API_KEY": "secret:public",
                "AIPLATFORM_BASE_URL": "https://aiapidev.3ecompany.com/",
                "AIPLATFORM_TIMEOUT": "30",
                "AIPLATFORM_CHAT_PATH": "v1/chat/completions",
                "MODEL": "baseten/deepseek-ai/deepseek-v3.1",
            },
            clear=True,
        ):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.llm_provider, "aiplatform")
        self.assertEqual(settings.aiplatform_api_key, "secret:public")
        self.assertEqual(settings.aiplatform_base_url, "https://aiapidev.3ecompany.com")
        self.assertEqual(settings.aiplatform_timeout, 30)
        self.assertEqual(settings.aiplatform_chat_path, "v1/chat/completions")
        self.assertEqual(settings.preferred_models, ("baseten/deepseek-ai/deepseek-v3.1",))

    def test_get_settings_rejects_non_numeric_request_timeout(self) -> None:
        with patch.dict(os.environ, {"WEEKEND_WIZARD_REQUEST_TIMEOUT": "slow"}, clear=True):
            get_settings.cache_clear()
            with self.assertRaisesRegex(ValueError, "WEEKEND_WIZARD_REQUEST_TIMEOUT"):
                get_settings()

    def test_get_settings_rejects_zero_or_negative_positive_timeouts(self) -> None:
        for name, value in (
            ("WEEKEND_WIZARD_REQUEST_TIMEOUT", "0"),
            ("WEEKEND_WIZARD_TOOL_HTTP_TIMEOUT", "-1"),
            ("WEEKEND_WIZARD_MAX_PROMPT_CHARS", "0"),
            ("WEEKEND_WIZARD_RATE_LIMIT_REQUESTS", "0"),
            ("WEEKEND_WIZARD_RATE_LIMIT_WINDOW_SECONDS", "0"),
        ):
            with self.subTest(name=name, value=value):
                with patch.dict(os.environ, {name: value}, clear=True):
                    get_settings.cache_clear()
                    with self.assertRaisesRegex(ValueError, name):
                        get_settings()

    def test_get_settings_rejects_negative_retry_values(self) -> None:
        for name, value in (
            ("WEEKEND_WIZARD_HTTP_MAX_RETRIES", "-1"),
            ("WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS", "-0.5"),
        ):
            with self.subTest(name=name, value=value):
                with patch.dict(os.environ, {name: value}, clear=True):
                    get_settings.cache_clear()
                    with self.assertRaisesRegex(ValueError, name):
                        get_settings()

    def test_get_settings_rejects_invalid_log_level(self) -> None:
        with patch.dict(os.environ, {"WEEKEND_WIZARD_LOG_LEVEL": "TRACE"}, clear=True):
            get_settings.cache_clear()
            with self.assertRaisesRegex(ValueError, "WEEKEND_WIZARD_LOG_LEVEL"):
                get_settings()

    def test_get_settings_rejects_invalid_llm_provider(self) -> None:
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}, clear=True):
            get_settings.cache_clear()
            with self.assertRaisesRegex(ValueError, "LLM_PROVIDER"):
                get_settings()


if __name__ == "__main__":
    unittest.main()
