from __future__ import annotations

import unittest

from guardrails.execution import ExecutionStateSnapshot, normalize_tool_args
from guardrails.guardrails import (
    infer_city,
    missing_requested_tools,
    parse_coords,
    requested_tools,
    validate_final_answer,
)
from schemas.agent import ExecutionPlan
from schemas.tools import BookItem, BookResults, DogResult, JokeResult, WeatherResult


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

    def test_requested_tools_does_not_infer_weather_from_city_alone(self) -> None:
        requested = requested_tools("Plan a weekend in Las Vegas with 3 adventure books.")

        self.assertEqual(requested, {"book_recs"})

    def test_requested_tools_does_not_infer_weather_from_coords_alone(self) -> None:
        requested = requested_tools("I'm at (40.7128, -74.0060). Give me 3 adventure books.")

        self.assertEqual(requested, {"book_recs"})

    def test_requested_tools_collects_multi_tool_prompt(self) -> None:
        requested = requested_tools(
            "Plan a weekend with weather, book ideas, a joke, a dog pic, and trivia."
        )

        self.assertEqual(
            requested,
            {"get_weather", "book_recs", "random_joke", "random_dog", "trivia"},
        )

    def test_requested_tools_does_not_infer_weather_from_photo_word(self) -> None:
        requested = requested_tools("Give me a dog photo.")

        self.assertEqual(requested, {"random_dog"})

    def test_normalize_book_args_rejects_legacy_param_key(self) -> None:
        normalized, error = normalize_tool_args(
            "book_recs",
            {"param": "mystery", "limit": 3},
            ExecutionStateSnapshot(
                plan=ExecutionPlan(goal="book_suggestions", execution_steps=[]),
                resolved_coords=None,
            ),
        )

        self.assertIsNone(normalized)
        self.assertEqual(error, "topic is required")

    def test_validate_final_answer_accepts_reflection_that_covers_requested_outputs(self) -> None:
        validation = validate_final_answer(
            "Plan a cozy Saturday with weather, 2 mystery books, a joke, and a dog pic.",
            (
                "Plan a cozy Saturday with clear sky and 20.5C weather. "
                "Read A Caribbean Mystery and The Mysterious Affair at Styles. "
                "Joke: Oysters hate to give away their pearls because they are shellfish. "
                "Dog: https://example.com/dog.jpg"
            ),
            {
                "get_weather": WeatherResult(temperature=20.5, temperature_unit="C", weather_summary="clear sky"),
                "book_recs": BookResults(
                    topic="mystery",
                    results=[
                        BookItem(title="A Caribbean Mystery", author="Agatha Christie"),
                        BookItem(title="The Mysterious Affair at Styles", author="Agatha Christie"),
                    ],
                ),
                "random_joke": JokeResult(
                    joke="Oysters hate to give away their pearls because they are shellfish."
                ),
                "random_dog": DogResult(image_url="https://example.com/dog.jpg"),
            },
        )

        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.missing_tools, ())

    def test_validate_final_answer_rejects_reflection_that_drops_requested_outputs(self) -> None:
        validation = validate_final_answer(
            "Plan a cozy Saturday with weather, 2 mystery books, a joke, and a dog pic.",
            "Plan a cozy Saturday in New York with clear sky and 20.5C weather.",
            {
                "get_weather": WeatherResult(temperature=20.5, temperature_unit="C", weather_summary="clear sky"),
                "book_recs": BookResults(
                    topic="mystery",
                    results=[BookItem(title="A Caribbean Mystery", author="Agatha Christie")],
                ),
                "random_joke": JokeResult(joke="A fetched joke."),
                "random_dog": DogResult(image_url="https://example.com/dog.jpg"),
            },
        )

        self.assertFalse(validation.is_valid)
        self.assertEqual(set(validation.missing_tools), {"book_recs", "random_dog"})

    def test_validate_final_answer_does_not_require_exact_joke_text(self) -> None:
        validation = validate_final_answer(
            "Tell me a joke.",
            "Here is a joke to brighten your day.",
            {
                "random_joke": JokeResult(
                    joke='Eight bytes walk into a bar. The bartender asks, "Can I get you anything?" "Yeah," reply the bytes. "Make us a double."'
                ),
            },
        )

        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.missing_tools, ())


if __name__ == "__main__":
    unittest.main()
