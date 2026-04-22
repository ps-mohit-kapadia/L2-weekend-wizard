from __future__ import annotations

import unittest

from agent.grounding import compose_grounded_answer_from_observations
from agent.prompts import build_react_messages, build_reflection_messages
from schemas.agent import ToolObservation


class PromptingTests(unittest.TestCase):
    def test_build_react_messages_include_decision_schema_and_tools(self) -> None:
        messages = build_react_messages(
            [{"role": "user", "content": "Plan a cozy Saturday in New York with weather and books."}],
            ["city_to_coords", "get_weather", "book_recs", "random_joke"],
            step_number=1,
            max_steps=6,
        )

        self.assertEqual(len(messages), 2)
        self.assertIn("ReAct-style weekend helper", messages[0]["content"])
        self.assertIn('"action":"tool"', messages[0]["content"])
        self.assertIn('"action":"finish"', messages[0]["content"])
        self.assertIn("step 1 of at most 6", messages[0]["content"])
        self.assertIn("Only call tools that are necessary", messages[0]["content"])
        self.assertIn("If the request is already satisfied", messages[0]["content"])
        self.assertIn('For "Tell me a joke.": call random_joke, then finish.', messages[0]["content"])
        self.assertIn("city_to_coords args", messages[0]["content"])
        self.assertIn("book_recs args", messages[0]["content"])
        self.assertIn("Plan a cozy Saturday", messages[1]["content"])

    def test_build_reflection_messages_include_observations_and_draft(self) -> None:
        messages = build_reflection_messages(
            "Tell me a joke.",
            [ToolObservation(tool_name="random_joke", args={}, payload='{"joke":"Hi"}')],
            "Joke: Hi",
        )

        self.assertEqual(len(messages), 2)
        self.assertIn('{"answer":"..."}', messages[0]["content"])
        self.assertIn("random_joke", messages[1]["content"])
        self.assertIn("Joke: Hi", messages[1]["content"])

    def test_compose_grounded_answer_returns_single_tool_fact(self) -> None:
        tool_observations = [
            ToolObservation(tool_name="random_joke", args={}, payload='{"joke": "A precise joke."}'),
        ]

        grounded = compose_grounded_answer_from_observations(
            "Tell me a joke.",
            "Placeholder answer.",
            tool_observations,
        )

        self.assertEqual(grounded, "Joke: A precise joke.")

    def test_compose_grounded_answer_prefers_fetched_facts_for_plan_requests(self) -> None:
        tool_observations = [
            ToolObservation(
                tool_name="city_to_coords",
                args={},
                payload='{"city": "New York", "latitude": 40.7128, "longitude": -74.0060, "country": "United States"}',
            ),
            ToolObservation(
                tool_name="get_weather",
                args={},
                payload='{"temperature": 4.0, "temperature_unit": "C", "weather_summary": "clear sky"}',
            ),
            ToolObservation(
                tool_name="book_recs",
                args={},
                payload='{"topic": "mystery", "results": [{"title": "A Caribbean Mystery", "author": "Agatha Christie"}, {"title": "The Mysterious Affair at Styles", "author": "Agatha Christie"}]}',
            ),
            ToolObservation(
                tool_name="random_joke",
                args={},
                payload='{"joke": "Fetched joke text."}',
            ),
            ToolObservation(
                tool_name="random_dog",
                args={},
                payload='{"image_url": "https://example.com/dog.jpg"}',
            ),
        ]

        composed = compose_grounded_answer_from_observations(
            "Plan a cozy Saturday in New York with weather, books, a joke, and a dog pic.",
            "Hallucinated answer here.",
            tool_observations,
        )

        self.assertTrue(composed.startswith("Weekend Wizard Plan"))
        self.assertIn("- Books: A Caribbean Mystery by Agatha Christie; The Mysterious Affair at Styles by Agatha Christie", composed)
        self.assertIn("- Joke: Fetched joke text.", composed)
        self.assertIn("- Dog Pic: https://example.com/dog.jpg", composed)
        self.assertNotIn("Hallucinated answer here.", composed)


if __name__ == "__main__":
    unittest.main()
