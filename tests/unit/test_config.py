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
                "WEEKEND_WIZARD_TOOL_HTTP_TIMEOUT": "12",
                "WEEKEND_WIZARD_HTTP_MAX_RETRIES": "4",
                "WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS": "0.25",
                "WEEKEND_WIZARD_LOG_LEVEL": "INFO",
                "WEEKEND_WIZARD_API_KEY": "secret-key",
                "WEEKEND_WIZARD_API_HOST": "0.0.0.0",
                "WEEKEND_WIZARD_API_PORT": "9000",
                "WEEKEND_WIZARD_API_URL": "http://0.0.0.0:9000",
                "WEEKEND_WIZARD_PREFERRED_MODELS": "gpt-oss:20b-cloud,llama3.1:8b",
                "WEEKEND_WIZARD_OBSERVABILITY_MODE": "local",
                "OLLAMA_URL": "http://localhost:11434/api/chat",
                "OLLAMA_TAGS_URL": "http://localhost:11434/api/tags",
            },
            clear=False,
        ):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.tool_http_timeout, 12)
        self.assertEqual(settings.http_max_retries, 4)
        self.assertEqual(settings.http_retry_backoff_seconds, 0.25)
        self.assertEqual(settings.log_level, "INFO")
        self.assertEqual(settings.api_key, "secret-key")
        self.assertEqual(settings.api_host, "0.0.0.0")
        self.assertEqual(settings.api_port, 9000)
        self.assertEqual(settings.api_url, "http://0.0.0.0:9000")
        self.assertEqual(settings.preferred_models, ("gpt-oss:20b-cloud", "llama3.1:8b"))
        self.assertEqual(settings.ollama_url, "http://localhost:11434/api/chat")
        self.assertEqual(settings.ollama_tags_url, "http://localhost:11434/api/tags")
        self.assertEqual(settings.observability_mode, "local")

    def test_get_settings_derives_ollama_tags_url_from_chat_url(self) -> None:
        with (
            patch.dict(
                os.environ,
                {
                    "OLLAMA_URL": "http://localhost:11434/api/chat",
                },
                clear=True,
            ),
            patch("config.config.Path.exists", return_value=False),
        ):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.ollama_tags_url, "http://localhost:11434/api/tags")

    def test_get_settings_reloads_when_environment_changes(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WEEKEND_WIZARD_TOOL_HTTP_TIMEOUT": "9",
                "WEEKEND_WIZARD_HTTP_MAX_RETRIES": "6",
            },
            clear=False,
        ):
            get_settings.cache_clear()
            settings = get_settings()

            self.assertEqual(settings.tool_http_timeout, 9)
            self.assertEqual(settings.http_max_retries, 6)

    def test_get_settings_loads_local_dotenv_values_when_not_present_in_environment(self) -> None:
        dotenv_content = "\n".join(
            [
                "WEEKEND_WIZARD_API_KEY=dotenv-secret",
                "WEEKEND_WIZARD_LOG_LEVEL=debug",
            ]
        )
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("config.config.Path.exists", return_value=True),
            patch("config.config.Path.read_text", return_value=dotenv_content),
        ):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.api_key, "dotenv-secret")
        self.assertEqual(settings.log_level, "DEBUG")

    def test_get_settings_normalizes_observability_mode(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WEEKEND_WIZARD_OBSERVABILITY_MODE": "Staging",
            },
            clear=False,
        ):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.observability_mode, "staging")

    def test_environment_values_override_local_dotenv(self) -> None:
        with (
            patch.dict(os.environ, {"WEEKEND_WIZARD_API_KEY": "env-secret"}, clear=True),
            patch("config.config.Path.exists", return_value=True),
            patch("config.config.Path.read_text", return_value="WEEKEND_WIZARD_API_KEY=dotenv-secret"),
        ):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.api_key, "env-secret")

    def test_get_settings_rejects_non_numeric_request_timeout(self) -> None:
        with patch.dict(os.environ, {"WEEKEND_WIZARD_REQUEST_TIMEOUT": "abc"}, clear=False):
            get_settings.cache_clear()
            with self.assertRaisesRegex(ValueError, "WEEKEND_WIZARD_REQUEST_TIMEOUT must be an integer."):
                get_settings()

    def test_get_settings_rejects_non_positive_timeouts(self) -> None:
        for env_name in ("WEEKEND_WIZARD_REQUEST_TIMEOUT", "WEEKEND_WIZARD_TOOL_HTTP_TIMEOUT"):
            with self.subTest(env_name=env_name):
                with patch.dict(os.environ, {env_name: "0"}, clear=False):
                    get_settings.cache_clear()
                    with self.assertRaisesRegex(ValueError, f"{env_name} must be positive."):
                        get_settings()

    def test_get_settings_rejects_negative_retry_count(self) -> None:
        with patch.dict(os.environ, {"WEEKEND_WIZARD_HTTP_MAX_RETRIES": "-1"}, clear=False):
            get_settings.cache_clear()
            with self.assertRaisesRegex(ValueError, "WEEKEND_WIZARD_HTTP_MAX_RETRIES must be zero or positive."):
                get_settings()

    def test_get_settings_allows_zero_retry_values(self) -> None:
        with patch.dict(
            os.environ,
            {
                "WEEKEND_WIZARD_HTTP_MAX_RETRIES": "0",
                "WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS": "0",
            },
            clear=False,
        ):
            get_settings.cache_clear()
            settings = get_settings()

        self.assertEqual(settings.http_max_retries, 0)
        self.assertEqual(settings.http_retry_backoff_seconds, 0.0)

    def test_get_settings_rejects_negative_retry_backoff(self) -> None:
        with patch.dict(os.environ, {"WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS": "-0.1"}, clear=False):
            get_settings.cache_clear()
            with self.assertRaisesRegex(
                ValueError,
                "WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS must be zero or positive.",
            ):
                get_settings()

    def test_get_settings_rejects_invalid_api_port(self) -> None:
        for value in ("abc", "0", "70000"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"WEEKEND_WIZARD_API_PORT": value}, clear=False):
                    get_settings.cache_clear()
                    with self.assertRaisesRegex(
                        ValueError,
                        "WEEKEND_WIZARD_API_PORT must be an integer from 1 to 65535.",
                    ):
                        get_settings()


if __name__ == "__main__":
    unittest.main()
