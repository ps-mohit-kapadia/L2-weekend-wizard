from __future__ import annotations

import unittest
from types import SimpleNamespace
import requests
from unittest.mock import Mock, patch

from tools.books import book_recs
from tools.shared import error_payload, get_json
from tools.weather import get_weather


class ToolTests(unittest.TestCase):
    @patch("tools.books.get_json")
    def test_book_recs_transforms_open_library_docs(self, mock_get_json: Mock) -> None:
        mock_get_json.return_value = {
            "docs": [
                {
                    "title": "Dune",
                    "author_name": ["Frank Herbert"],
                    "first_publish_year": 1965,
                    "key": "/works/OL1W",
                }
            ]
        }

        result = book_recs("sci-fi", limit=1)

        self.assertEqual(result["topic"], "sci-fi")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["title"], "Dune")

    @patch("tools.weather.get_json")
    def test_get_weather_maps_summary_fields(self, mock_get_json: Mock) -> None:
        mock_get_json.return_value = {
            "current": {
                "time": "2026-04-09T12:00",
                "temperature_2m": 22.5,
                "weather_code": 1,
                "wind_speed_10m": 9.1,
            },
            "current_units": {
                "temperature_2m": "C",
                "wind_speed_10m": "km/h",
            },
        }

        result = get_weather(12.97, 77.59)

        self.assertEqual(result["temperature"], 22.5)
        self.assertEqual(result["weather_summary"], "mainly clear")
        self.assertEqual(result["wind_speed_unit"], "km/h")

    def test_error_payload_returns_consistent_shape(self) -> None:
        payload = error_payload("weather", RuntimeError("boom"))

        self.assertEqual(payload["error"], "weather request failed")
        self.assertIn("boom", payload["details"])

    @patch("tools.shared.time.sleep", return_value=None)
    @patch("tools.shared.requests.get")
    @patch("tools.shared.get_settings")
    def test_get_json_retries_and_recovers(self, mock_settings: Mock, mock_get: Mock, _sleep: Mock) -> None:
        mock_settings.return_value = SimpleNamespace(
            tool_http_timeout=22,
            http_max_retries=1,
            http_retry_backoff_seconds=0.5,
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"ok": True}

        mock_get.side_effect = [
            requests.RequestException("temporary network issue"),
            response,
        ]

        result = get_json("https://example.com")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_get.call_args.kwargs["timeout"], 22)

    @patch("tools.shared.time.sleep", return_value=None)
    @patch("tools.shared.requests.get")
    @patch("tools.shared.get_settings")
    def test_get_json_raises_after_retry_exhaustion(self, mock_settings: Mock, mock_get: Mock, _sleep: Mock) -> None:
        mock_settings.return_value = SimpleNamespace(
            tool_http_timeout=9,
            http_max_retries=0,
            http_retry_backoff_seconds=0.5,
        )
        mock_get.side_effect = requests.RequestException("still failing")

        with self.assertRaises(requests.RequestException):
            get_json("https://example.com")


if __name__ == "__main__":
    unittest.main()
