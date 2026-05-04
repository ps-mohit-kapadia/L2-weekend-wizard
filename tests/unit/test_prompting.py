from __future__ import annotations

import unittest

from agent.grounding import build_grounded_draft_from_payloads
from agent.prompts import build_planner_messages, build_reflection_messages
from schemas.tools import BookItem, BookResults, DogResult, GeoResult, JokeResult, WeatherResult


class PromptingTests(unittest.TestCase):
    def test_build_planner_messages_include_schema_and_tools(self) -> None:
        messages = build_planner_messages(
            "Plan a cozy Saturday in New York with weather and books.",
            ["city_to_coords", "get_weather", "book_recs", "random_joke"],
        )

        self.assertEqual(len(messages), 2)
        self.assertIn("structured execution plan", messages[0]["content"])
        self.assertIn("execution_steps", messages[0]["content"])
        self.assertIn("Choose exactly one goal value", messages[0]["content"])
        self.assertIn('"goal":"weekend_plan"', messages[0]["content"])
        self.assertIn('"goal":"trivia"', messages[0]["content"])
        self.assertIn('{"tool":"trivia","args":{}}', messages[0]["content"])
        self.assertIn("If the user asks only for trivia, plan only trivia.", messages[0]["content"])
        self.assertIn("Do not include trivia unless the user explicitly asked for trivia.", messages[0]["content"])
        self.assertIn("city_to_coords args", messages[0]["content"])
        self.assertIn("book_recs args", messages[0]["content"])
        self.assertIn("Plan a cozy Saturday", messages[1]["content"])

    def test_build_reflection_messages_include_observations_and_draft(self) -> None:
        messages = build_reflection_messages(
            "Tell me a joke.",
            {"random_joke": JokeResult(joke="Hi")},
            "Joke: Hi",
        )

        self.assertEqual(len(messages), 2)
        self.assertIn('{"answer":"..."}', messages[0]["content"])
        self.assertIn("preserve exact fetched book titles", messages[0]["content"])
        self.assertIn("exact joke text", messages[0]["content"])
        self.assertIn("exact dog URL", messages[0]["content"])
        self.assertIn("smooth natural prose", messages[0]["content"])
        self.assertIn("do not keep labels like Weather:", messages[0]["content"])
        self.assertIn("random_joke", messages[1]["content"])
        self.assertIn("Joke: Hi", messages[1]["content"])

    def test_compose_grounded_answer_returns_single_tool_fact(self) -> None:
        grounded = build_grounded_draft_from_payloads(
            "Tell me a joke.",
            "Placeholder answer.",
            {"random_joke": JokeResult(joke="A precise joke.")},
        )

        self.assertEqual(grounded, "Joke: A precise joke.")

    def test_compose_grounded_answer_prefers_fetched_facts_for_plan_requests(self) -> None:
        composed = build_grounded_draft_from_payloads(
            "Plan a cozy Saturday in New York with weather, books, a joke, and a dog pic.",
            "Hallucinated answer here.",
            {
                "city_to_coords": GeoResult(
                    city="New York",
                    latitude=40.7128,
                    longitude=-74.0060,
                    country="United States",
                ),
                "get_weather": WeatherResult(
                    temperature=4.0,
                    temperature_unit="C",
                    weather_summary="clear sky",
                ),
                "book_recs": BookResults(
                    topic="mystery",
                    results=[
                        BookItem(title="A Caribbean Mystery", author="Agatha Christie"),
                        BookItem(title="The Mysterious Affair at Styles", author="Agatha Christie"),
                        BookItem(title="Murder on the Orient Express", author="Agatha Christie"),
                    ],
                ),
                "random_joke": JokeResult(joke="Fetched joke text."),
                "random_dog": DogResult(image_url="https://example.com/dog.jpg"),
            },
        )

        self.assertTrue(composed.startswith("Weekend Wizard Plan"))
        self.assertIn(
            "- Books: A Caribbean Mystery by Agatha Christie; The Mysterious Affair at Styles by Agatha Christie; Murder on the Orient Express by Agatha Christie",
            composed,
        )
        self.assertIn("- Joke: Fetched joke text.", composed)
        self.assertIn("- Dog Pic: https://example.com/dog.jpg", composed)
        self.assertNotIn("Hallucinated answer here.", composed)


if __name__ == "__main__":
    unittest.main()
