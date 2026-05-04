from __future__ import annotations

import unittest
import requests
from unittest.mock import AsyncMock, Mock, patch

from tools.books import book_recs
from tools.shared import error_payload, get_json
from tools.weather import get_weather


class ToolTests(unittest.IsolatedAsyncioTestCase):
    @patch("tools.books.get_json", new_callable=AsyncMock)
    async def test_book_recs_transforms_open_library_docs(self, mock_get_json: AsyncMock) -> None:
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

        result = await book_recs("sci-fi", limit=1)

        self.assertEqual(result["topic"], "sci-fi")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["results"][0]["title"], "Dune")

    @patch("tools.weather.get_json", new_callable=AsyncMock)
    async def test_get_weather_maps_summary_fields(self, mock_get_json: AsyncMock) -> None:
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

        result = await get_weather(12.97, 77.59)

        self.assertEqual(result["temperature"], 22.5)
        self.assertEqual(result["weather_summary"], "mainly clear")
        self.assertEqual(result["wind_speed_unit"], "km/h")

    def test_error_payload_returns_consistent_shape(self) -> None:
        payload = error_payload("weather", RuntimeError("boom"))

        self.assertEqual(payload["error"], "weather request failed")
        self.assertIn("boom", payload["details"])

    @patch("tools.shared.asyncio.sleep", new_callable=AsyncMock)
    @patch("tools.shared.requests.get")
    async def test_get_json_retries_and_recovers(self, mock_get: Mock, mock_sleep: AsyncMock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"ok": True}

        mock_get.side_effect = [
            requests.RequestException("temporary network issue"),
            response,
        ]

        result = await get_json("https://example.com")

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_get.call_count, 2)
        mock_sleep.assert_awaited_once()

    @patch("tools.shared.asyncio.sleep", new_callable=AsyncMock)
    @patch("tools.shared.requests.get")
    async def test_get_json_raises_after_retry_exhaustion(self, mock_get: Mock, mock_sleep: AsyncMock) -> None:
        mock_get.side_effect = requests.RequestException("still failing")

        with self.assertRaises(requests.RequestException):
            await get_json("https://example.com")
        self.assertGreaterEqual(mock_sleep.await_count, 1)


if __name__ == "__main__":
    unittest.main()
