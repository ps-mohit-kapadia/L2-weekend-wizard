from __future__ import annotations

import unittest

from guardrails.guardrails import (
    analyze_request,
    infer_city,
    missing_requested_tools,
    parse_coords,
    requested_tools,
)


class PolicyTests(unittest.TestCase):
    def test_parse_coords_extracts_coordinate_tuple(self) -> None:
        coords = parse_coords("What's the weather at (40.7128, -74.0060)?")

        self.assertEqual(coords, (40.7128, -74.006))

    def test_parse_coords_extracts_plain_coordinate_tuple(self) -> None:
        coords = parse_coords("I'm in 40.7128, -74.0060 and want a weekend plan.")

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

    def test_analyze_request_builds_deterministic_plan_for_city_prompt(self) -> None:
        analysis = analyze_request(
            "Plan a cozy Saturday in New York with today's weather, 3 cozy mystery book ideas, a joke, and a dog pic.",
            ["city_to_coords", "get_weather", "book_recs", "random_joke", "random_dog"],
        )

        self.assertIsNotNone(analysis)
        assert analysis is not None
        self.assertEqual(
            analysis.requested_tools,
            ("get_weather", "book_recs", "random_joke", "random_dog"),
        )
        self.assertEqual(analysis.city, "New York")
        self.assertIsNone(analysis.coords)
        self.assertEqual(analysis.book_topic, "cozy mystery")
        self.assertEqual(analysis.book_limit, 3)

    def test_analyze_request_returns_none_when_weather_has_no_location(self) -> None:
        analysis = analyze_request(
            "What's the weather like today with a joke?",
            ["get_weather", "random_joke"],
        )

        self.assertIsNone(analysis)


if __name__ == "__main__":
    unittest.main()
