from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.grounding import (
    compose_grounded_answer_from_steps,
    render_reflection_observations,
)
from agent.policies.guardrails import analyze_request
from agent.prompts import build_react_messages, build_reflection_messages
from schemas.tools import BookResults, DogResult, GeoResult, JokeResult, ToolError, WeatherResult


class PromptingTests(unittest.TestCase):
    def test_build_react_messages_include_decision_schema_and_tools(self) -> None:
        messages = build_react_messages(
            [{"role": "user", "content": "Plan a cozy Saturday in New York with weather and books."}],
            ["city_to_coords", "get_weather", "book_recs", "random_joke"],
            step_number=1,
            max_steps=6,
            request_analysis=analyze_request(
                "Plan a cozy Saturday in New York with weather and books.",
                ["city_to_coords", "get_weather", "book_recs", "random_joke"],
            ),
        )

        self.assertEqual(len(messages), 2)
        self.assertIn("ReAct-style weekend helper", messages[0]["content"])
        self.assertIn('"action":"tool"', messages[0]["content"])
        self.assertIn('"action":"finish"', messages[0]["content"])
        self.assertIn("step 1 of at most 6", messages[0]["content"])
        self.assertIn("Only call tools that are necessary", messages[0]["content"])
        self.assertIn("Never introduce a new city, topic, location, or target", messages[0]["content"])
        self.assertIn("Tools gather external facts only.", messages[0]["content"])
        self.assertIn('action="finish" using final_answer', messages[0]["content"])
        self.assertIn("Only call one of the listed supported tools.", messages[0]["content"])
        self.assertIn("Never invent tool names.", messages[0]["content"])
        self.assertIn("If the request is already satisfied", messages[0]["content"])
        self.assertIn("Single-shot tools are random_joke, random_dog, and trivia.", messages[0]["content"])
        self.assertIn('For "Tell me a joke.": call random_joke once, then finish.', messages[0]["content"])
        self.assertIn('For "Give me a trivia question.": call trivia once, then finish.', messages[0]["content"])
        self.assertIn('For "Get the weather for City A and City B.": fetch the needed weather results, then finish.', messages[0]["content"])
        self.assertIn("Use the interpreted request facts below as the source of truth", messages[0]["content"])
        self.assertIn("Interpreted request facts:", messages[0]["content"])
        self.assertIn("- requested tools:", messages[0]["content"])
        self.assertIn("get_weather", messages[0]["content"])
        self.assertIn("book_recs", messages[0]["content"])
        self.assertIn("- provided city: New York", messages[0]["content"])
        self.assertNotIn("dependency fact:", messages[0]["content"])
        self.assertNotIn('Get the weather for Chicago and New York using their coordinates.', messages[0]["content"])
        self.assertNotIn("each requested location needs its own get_weather result before finish", messages[0]["content"])
        self.assertNotIn("Do not finish while any requested or already-resolved location still lacks a weather observation", messages[0]["content"])
        self.assertNotIn("Use weather only if the user asked for weather", messages[0]["content"])
        self.assertIn("city_to_coords args", messages[0]["content"])
        self.assertIn("book_recs args", messages[0]["content"])
        self.assertIn("Plan a cozy Saturday", messages[1]["content"])

    def test_build_react_messages_preserve_planner_transcript_roles_and_order(self) -> None:
        messages = build_react_messages(
            [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Thought: fetch one joke"},
                {"role": "tool", "content": "- Joke: A fetched joke."},
            ],
            ["random_joke"],
            step_number=2,
            max_steps=6,
        )

        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "Tell me a joke.")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertIn("fetch one joke", messages[2]["content"])
        self.assertEqual(messages[3]["role"], "tool")
        self.assertIn("A fetched joke.", messages[3]["content"])

    def test_build_reflection_messages_include_observations_and_draft(self) -> None:
        step_summary_lines = render_reflection_observations(
            "Tell me a joke.",
            [
                SimpleNamespace(
                    kind="tool_call",
                    tool_name="random_joke",
                    normalized_args={},
                    parsed_payload=JokeResult(joke="Hi"),
                )
            ],
        )
        messages = build_reflection_messages(
            "Tell me a joke.",
            step_summary_lines,
            "Joke: Hi",
        )

        self.assertEqual(len(messages), 2)
        self.assertIn('{"verdict":"pass","intro":"...","outro":"...","issues":[]}', messages[0]["content"])
        self.assertIn("Do not rewrite the grounded facts", messages[0]["content"])
        self.assertIn("optional intro and outro", messages[0]["content"])
        self.assertIn("- Joke: Hi", messages[1]["content"])
        self.assertIn("Joke: Hi", messages[1]["content"])
        self.assertNotIn('{"joke":"Hi"}', messages[1]["content"])

    def test_build_reflection_messages_use_compact_error_detail_without_raw_payload_blob(self) -> None:
        step_summary_lines = render_reflection_observations(
            "Give me the weather and a joke.",
            [
                SimpleNamespace(
                    kind="tool_call",
                    tool_name="get_weather",
                    normalized_args={"latitude": 40.7128, "longitude": -74.0060},
                    parsed_payload=ToolError(
                        error="get_weather failed",
                        details="tool execution failed",
                    ),
                ),
                SimpleNamespace(
                    kind="tool_call",
                    tool_name="random_joke",
                    normalized_args={},
                    parsed_payload=JokeResult(joke="Hi"),
                ),
            ],
        )
        messages = build_reflection_messages(
            "Give me the weather and a joke.",
            step_summary_lines,
            "Weather failed, but here is a joke.",
        )

        self.assertIn("- Weather: 40.7128, -74.006 unavailable (tool execution failed)", messages[1]["content"])
        self.assertIn("- Joke: Hi", messages[1]["content"])
        self.assertNotIn('{"error":"get_weather failed"', messages[1]["content"])
        self.assertNotIn('{"joke":"Hi"}', messages[1]["content"])

    def test_build_reflection_messages_include_grounded_dog_url_without_raw_payload_blob(self) -> None:
        step_summary_lines = render_reflection_observations(
            "Plan a cozy Saturday with a dog pic.",
            [
                SimpleNamespace(
                    kind="tool_call",
                    tool_name="random_dog",
                    normalized_args={},
                    parsed_payload=DogResult(
                        status="success",
                        image_url="https://example.com/dog.jpg",
                    ),
                ),
            ],
        )
        messages = build_reflection_messages(
            "Plan a cozy Saturday with a dog pic.",
            step_summary_lines,
            "Here is a dog pic.",
        )

        self.assertIn("- Dog Pic: https://example.com/dog.jpg", messages[1]["content"])
        self.assertNotIn('{"status":"success","image_url"', messages[1]["content"])

    def test_compose_grounded_answer_returns_single_tool_fact(self) -> None:
        steps = [
            SimpleNamespace(
                kind="tool_call",
                tool_name="random_joke",
                normalized_args={},
                parsed_payload=JokeResult(joke="A precise joke."),
            ),
        ]

        grounded = compose_grounded_answer_from_steps(
            "Tell me a joke.",
            "Placeholder answer.",
            steps,
        )

        self.assertEqual(grounded, "Joke: A precise joke.")

    def test_compose_grounded_answer_prefers_fetched_facts_for_plan_requests(self) -> None:
        steps = [
            SimpleNamespace(
                kind="tool_call",
                tool_name="city_to_coords",
                normalized_args={},
                parsed_payload=GeoResult(
                    city="New York",
                    latitude=40.7128,
                    longitude=-74.0060,
                    country="United States",
                ),
            ),
            SimpleNamespace(
                kind="tool_call",
                tool_name="get_weather",
                normalized_args={},
                parsed_payload=WeatherResult(
                    latitude=40.7128,
                    longitude=-74.0060,
                    temperature=4.0,
                    temperature_unit="C",
                    weather_summary="clear sky",
                ),
            ),
            SimpleNamespace(
                kind="tool_call",
                tool_name="book_recs",
                normalized_args={},
                parsed_payload=BookResults(
                    topic="mystery",
                    count=2,
                    results=[
                        {"title": "A Caribbean Mystery", "author": "Agatha Christie"},
                        {"title": "The Mysterious Affair at Styles", "author": "Agatha Christie"},
                    ],
                ),
            ),
            SimpleNamespace(
                kind="tool_call",
                tool_name="random_joke",
                normalized_args={},
                parsed_payload=JokeResult(joke="Fetched joke text."),
            ),
            SimpleNamespace(
                kind="tool_call",
                tool_name="random_dog",
                normalized_args={},
                parsed_payload=DogResult(status="success", image_url="https://example.com/dog.jpg"),
            ),
        ]

        composed = compose_grounded_answer_from_steps(
            "Plan a cozy Saturday in New York with weather, books, a joke, and a dog pic.",
            "Hallucinated answer here.",
            steps,
        )

        self.assertTrue(composed.startswith("Weekend Wizard Plan"))
        self.assertIn("- City Lookup: New York: 40.7128, -74.006", composed)
        self.assertIn("- Weather: requested location: 4.0C, clear sky", composed)
        self.assertIn("- Books: A Caribbean Mystery by Agatha Christie; The Mysterious Affair at Styles by Agatha Christie", composed)
        self.assertIn("- Joke: Fetched joke text.", composed)
        self.assertIn("- Dog Pic: https://example.com/dog.jpg", composed)
        self.assertNotIn("Hallucinated answer here.", composed)

    def test_compose_grounded_answer_preserves_two_weather_observations(self) -> None:
        steps = [
            SimpleNamespace(
                kind="tool_call",
                tool_name="get_weather",
                normalized_args={"latitude": 41.85003, "longitude": -87.65005},
                parsed_payload=WeatherResult(
                    latitude=41.85003,
                    longitude=-87.65005,
                    temperature=11.2,
                    temperature_unit="C",
                    weather_summary="clear sky",
                ),
            ),
            SimpleNamespace(
                kind="tool_call",
                tool_name="get_weather",
                normalized_args={"latitude": 40.71427, "longitude": -74.00597},
                parsed_payload=WeatherResult(
                    latitude=40.71427,
                    longitude=-74.00597,
                    temperature=6.1,
                    temperature_unit="C",
                    weather_summary="light rain",
                ),
            ),
        ]

        composed = compose_grounded_answer_from_steps(
            "Compare the weather in Chicago and New York.",
            "Placeholder answer.",
            steps,
        )

        self.assertIn("Weekend Wizard Results", composed)
        self.assertIn("- Weather: 41.85003, -87.65005: 11.2C, clear sky", composed)
        self.assertIn("- Weather: 40.71427, -74.00597: 6.1C, light rain", composed)

    def test_build_reflection_messages_preserve_two_weather_observations(self) -> None:
        step_summary_lines = render_reflection_observations(
            "Compare the weather in Chicago and New York.",
            [
                SimpleNamespace(
                    kind="tool_call",
                    tool_name="get_weather",
                    normalized_args={"latitude": 41.85003, "longitude": -87.65005},
                    parsed_payload=WeatherResult(
                        latitude=41.85003,
                        longitude=-87.65005,
                        temperature=11.2,
                        temperature_unit="C",
                        weather_summary="clear sky",
                    ),
                ),
                SimpleNamespace(
                    kind="tool_call",
                    tool_name="get_weather",
                    normalized_args={"latitude": 40.71427, "longitude": -74.00597},
                    parsed_payload=WeatherResult(
                        latitude=40.71427,
                        longitude=-74.00597,
                        temperature=6.1,
                        temperature_unit="C",
                        weather_summary="light rain",
                    ),
                ),
            ],
        )
        messages = build_reflection_messages(
            "Compare the weather in Chicago and New York.",
            step_summary_lines,
            "Draft answer.",
        )

        self.assertIn("- Weather: 41.85003, -87.65005: 11.2C, clear sky", messages[1]["content"])
        self.assertIn("- Weather: 40.71427, -74.00597: 6.1C, light rain", messages[1]["content"])

    def test_build_reflection_messages_preserve_two_city_lookups(self) -> None:
        step_summary_lines = render_reflection_observations(
            "Compare Chicago and New York.",
            [
                SimpleNamespace(
                    kind="tool_call",
                    tool_name="city_to_coords",
                    normalized_args={"city": "Chicago"},
                    parsed_payload=GeoResult(
                        city="Chicago",
                        latitude=41.85003,
                        longitude=-87.65005,
                        country="United States",
                    ),
                ),
                SimpleNamespace(
                    kind="tool_call",
                    tool_name="city_to_coords",
                    normalized_args={"city": "New York"},
                    parsed_payload=GeoResult(
                        city="New York",
                        latitude=40.71427,
                        longitude=-74.00597,
                        country="United States",
                    ),
                ),
            ],
        )
        messages = build_reflection_messages(
            "Compare Chicago and New York.",
            step_summary_lines,
            "Draft answer.",
        )

        self.assertIn("- City Lookup: Chicago: 41.85003, -87.65005", messages[1]["content"])
        self.assertIn("- City Lookup: New York: 40.71427, -74.00597", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
