from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.grounding import compose_grounded_answer_from_observations
from agent.prompts import build_system_prompt
from schemas.agent import ToolObservation


class PromptingTests(unittest.TestCase):
    def test_build_system_prompt_lists_available_tools(self) -> None:
        tools = [
            SimpleNamespace(
                name="get_weather",
                description="Fetches weather",
                inputSchema={"type": "object", "properties": {"latitude": {"type": "number"}}},
            )
        ]

        prompt = build_system_prompt(tools)

        self.assertIn("Weekend Wizard", prompt)
        self.assertIn("get_weather", prompt)
        self.assertIn("Fetches weather", prompt)

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
