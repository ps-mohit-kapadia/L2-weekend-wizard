from __future__ import annotations

import unittest

from agent.policies.guardrails import (
    infer_city,
    missing_requested_tools,
    parse_coords,
    requested_tools,
)


class PolicyTests(unittest.TestCase):
    def test_parse_coords_extracts_coordinate_tuple(self) -> None:
        coords = parse_coords("What's the weather at (40.7128, -74.0060)?")

        self.assertEqual(coords, (40.7128, -74.006))

    def test_missing_requested_tools_identifies_unfetched_requests(self) -> None:
        payloads = {
            "get_weather": {"temperature": 21},
            "book_recs": {"results": []},
        }

        missing = missing_requested_tools(
            "Plan a weekend with weather, book ideas, a joke, and a dog pic.",
            payloads,
        )

        self.assertEqual(set(missing), {"random_joke", "random_dog"})

    def test_infer_city_extracts_capitalized_city_phrase(self) -> None:
        city = infer_city("Plan a cozy Saturday in New York with mystery books.")

        self.assertEqual(city, "New York")

    def test_requested_tools_includes_weather_for_city_prompt(self) -> None:
        requested = requested_tools("What's the weather in New York? Keep it brief.")

        self.assertIn("get_weather", requested)

    def test_requested_tools_collects_multi_tool_prompt(self) -> None:
        requested = requested_tools(
            "Plan a weekend with weather, book ideas, a joke, a dog pic, and trivia."
        )

        self.assertEqual(
            requested,
            {"get_weather", "book_recs", "random_joke", "random_dog", "trivia"},
        )


if __name__ == "__main__":
    unittest.main()
