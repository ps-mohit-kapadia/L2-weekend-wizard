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
        self.assertIn("Single-shot tools are random_joke, random_dog, and trivia.", messages[0]["content"])
        self.assertIn('For "Tell me a joke.": call random_joke once, then finish.', messages[0]["content"])
        self.assertIn('For "Give me a trivia question.": call trivia once, then finish.', messages[0]["content"])
        self.assertIn("city_to_coords args", messages[0]["content"])
        self.assertIn("book_recs args", messages[0]["content"])
        self.assertIn("Plan a cozy Saturday", messages[1]["content"])

    def test_build_react_messages_include_compact_observation_summary_without_raw_payload(self) -> None:
        messages = build_react_messages(
            [{"role": "user", "content": "Tell me a joke."}],
            ["random_joke"],
            step_number=2,
            max_steps=6,
            observation_summary="- random_joke: fetched one joke",
        )

        self.assertEqual(len(messages), 3)
        self.assertIn("Structured observation summary", messages[1]["content"])
        self.assertIn("fetched one joke", messages[1]["content"])
        self.assertNotIn('{"joke"', messages[1]["content"])
        self.assertIn("Tell me a joke.", messages[2]["content"])

    def test_build_reflection_messages_include_observations_and_draft(self) -> None:
        messages = build_reflection_messages(
            "Tell me a joke.",
            [ToolObservation(tool_name="random_joke", args={}, payload='{"joke":"Hi"}')],
            "Joke: Hi",
        )

        self.assertEqual(len(messages), 2)
        self.assertIn('{"answer":"..."}', messages[0]["content"])
        self.assertIn("- Joke: Hi", messages[1]["content"])
        self.assertIn("Joke: Hi", messages[1]["content"])
        self.assertNotIn('{"joke":"Hi"}', messages[1]["content"])

    def test_build_reflection_messages_use_compact_error_detail_without_raw_payload_blob(self) -> None:
        messages = build_reflection_messages(
            "Give me the weather and a joke.",
            [
                ToolObservation(
                    tool_name="get_weather",
                    args={"latitude": 40.7128, "longitude": -74.0060},
                    payload='{"error":"get_weather failed","details":"tool execution failed"}',
                ),
                ToolObservation(tool_name="random_joke", args={}, payload='{"joke":"Hi"}'),
            ],
            "Weather failed, but here is a joke.",
        )

        self.assertIn("- Weather: 40.7128, -74.006 unavailable (tool execution failed)", messages[1]["content"])
        self.assertIn("- Joke: Hi", messages[1]["content"])
        self.assertNotIn('{"error":"get_weather failed"', messages[1]["content"])
        self.assertNotIn('{"joke":"Hi"}', messages[1]["content"])

    def test_build_reflection_messages_avoid_raw_url_payload_rendering(self) -> None:
        messages = build_reflection_messages(
            "Plan a cozy Saturday with a dog pic.",
            [
                ToolObservation(
                    tool_name="random_dog",
                    args={},
                    payload='{"status":"success","image_url":"https://example.com/dog.jpg"}',
                ),
            ],
            "Here is a dog pic.",
        )

        self.assertIn("- Dog Pic: fetched one dog image", messages[1]["content"])
        self.assertNotIn('{"status":"success","image_url"', messages[1]["content"])
        self.assertNotIn("https://example.com/dog.jpg", messages[1]["content"])

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
        self.assertIn("- City Lookup: New York: 40.7128, -74.006", composed)
        self.assertIn("- Weather: requested location: 4.0C, clear sky", composed)
        self.assertIn("- Books: A Caribbean Mystery by Agatha Christie; The Mysterious Affair at Styles by Agatha Christie", composed)
        self.assertIn("- Joke: Fetched joke text.", composed)
        self.assertIn("- Dog Pic: https://example.com/dog.jpg", composed)
        self.assertNotIn("Hallucinated answer here.", composed)

    def test_compose_grounded_answer_preserves_two_weather_observations(self) -> None:
        tool_observations = [
            ToolObservation(
                tool_name="get_weather",
                args={"latitude": 41.85003, "longitude": -87.65005},
                payload='{"temperature": 11.2, "temperature_unit": "C", "weather_summary": "clear sky"}',
            ),
            ToolObservation(
                tool_name="get_weather",
                args={"latitude": 40.71427, "longitude": -74.00597},
                payload='{"temperature": 6.1, "temperature_unit": "C", "weather_summary": "light rain"}',
            ),
        ]

        composed = compose_grounded_answer_from_observations(
            "Compare the weather in Chicago and New York.",
            "Placeholder answer.",
            tool_observations,
        )

        self.assertIn("Weekend Wizard Results", composed)
        self.assertIn("- Weather: 41.85003, -87.65005: 11.2C, clear sky", composed)
        self.assertIn("- Weather: 40.71427, -74.00597: 6.1C, light rain", composed)

    def test_build_reflection_messages_preserve_two_weather_observations(self) -> None:
        messages = build_reflection_messages(
            "Compare the weather in Chicago and New York.",
            [
                ToolObservation(
                    tool_name="get_weather",
                    args={"latitude": 41.85003, "longitude": -87.65005},
                    payload='{"temperature": 11.2, "temperature_unit": "C", "weather_summary": "clear sky"}',
                ),
                ToolObservation(
                    tool_name="get_weather",
                    args={"latitude": 40.71427, "longitude": -74.00597},
                    payload='{"temperature": 6.1, "temperature_unit": "C", "weather_summary": "light rain"}',
                ),
            ],
            "Draft answer.",
        )

        self.assertIn("- Weather: 41.85003, -87.65005: 11.2C, clear sky", messages[1]["content"])
        self.assertIn("- Weather: 40.71427, -74.00597: 6.1C, light rain", messages[1]["content"])

    def test_build_reflection_messages_preserve_two_city_lookups(self) -> None:
        messages = build_reflection_messages(
            "Compare Chicago and New York.",
            [
                ToolObservation(
                    tool_name="city_to_coords",
                    args={"city": "Chicago"},
                    payload='{"city": "Chicago", "latitude": 41.85003, "longitude": -87.65005, "country": "United States"}',
                ),
                ToolObservation(
                    tool_name="city_to_coords",
                    args={"city": "New York"},
                    payload='{"city": "New York", "latitude": 40.71427, "longitude": -74.00597, "country": "United States"}',
                ),
            ],
            "Draft answer.",
        )

        self.assertIn("- City Lookup: Chicago: 41.85003, -87.65005", messages[1]["content"])
        self.assertIn("- City Lookup: New York: 40.71427, -74.00597", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
