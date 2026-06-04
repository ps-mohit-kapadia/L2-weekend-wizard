from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent.prompts import build_react_messages
from agent.orchestrator import (
    SAFE_TOOL_INVOCATION_DETAIL,
    orchestrate_interaction,
    validate_react_decision_semantics,
)
from mcp_runtime.client import ToolInvocationError
from schemas.agent import OrchestratorContext, validate_react_decision


def fake_tool_result(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])


class OrchestratorIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def test_react_prompt_keeps_weather_as_capability_guidance_not_completion_owner(self) -> None:
        messages = build_react_messages(
            planner_messages=[],
            tool_names=["city_to_coords", "get_weather"],
            step_number=1,
            max_steps=6,
        )

        system_prompt = messages[0]["content"]
        self.assertIn("Never introduce a new city, topic, location, or target that the user did not ask for.", system_prompt)
        self.assertIn("Tools gather external facts only.", system_prompt)
        self.assertIn('Comparisons, summaries, recommendations, and final wording must happen in action="finish" using final_answer.', system_prompt)
        self.assertIn("Only call one of the listed supported tools.", system_prompt)
        self.assertIn("Never invent tool names.", system_prompt)
        self.assertIn("If weather is requested and coordinates are already available, prefer get_weather directly.", system_prompt)
        self.assertIn("If weather is requested and only a city is known, use city_to_coords before get_weather.", system_prompt)
        self.assertIn("If an identical successful tool call already appears in observations, do not request it again; choose a different needed step or finish.", system_prompt)
        self.assertIn('Get the weather for City A and City B.', system_prompt)
        self.assertNotIn('Get the weather for Chicago and New York using their coordinates.', system_prompt)
        self.assertNotIn("each requested location needs its own get_weather result before finish", system_prompt)
        self.assertNotIn("Resolving a city to coordinates is only a dependency, not fulfillment", system_prompt)
        self.assertNotIn("Do not finish while any requested or already-resolved location still lacks a weather observation", system_prompt)

    @patch(
        "agent.orchestrator.llm_reflection_json",
        return_value={
            "answer": (
                "Weekend Wizard Plan\n"
                "- Weather: 6.1C, clear sky\n"
                "- Books: A Caribbean Mystery by Agatha Christie\n"
                "- Joke: A fetched joke.\n"
                "- Dog Pic: https://example.com/dog.jpg"
            )
        },
    )
    @patch("agent.orchestrator.llm_react_json")
    async def test_city_prompt_flows_to_reflected_grounded_final_answer(
        self,
        mock_react: Mock,
        mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Need coordinates first.", "action": "tool", "tool": "city_to_coords", "args": {"city": "New York"}},
            {"thought": "Now get weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.71427, "longitude": -74.00597}},
            {"thought": "Need books.", "action": "tool", "tool": "book_recs", "args": {"param": "mystery", "limit": 2}},
            {"thought": "Need one joke.", "action": "tool", "tool": "random_joke", "args": {}},
            {"thought": "Need one dog photo.", "action": "tool", "tool": "random_dog", "args": {}},
            {"thought": "I have enough information.", "action": "finish", "final_answer": "Here is your cozy Saturday plan."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "city": "New York",
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "country": "United States",
                    "admin1": "New York",
                    "timezone": "America/New_York",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "temperature": 6.1,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
            fake_tool_result(
                {
                    "topic": "mystery",
                    "count": 2,
                    "results": [
                        {"title": "A Caribbean Mystery", "author": "Agatha Christie"},
                        {"title": "The Mysterious Affair at Styles", "author": "Agatha Christie"},
                    ],
                }
            ),
            fake_tool_result({"joke": "A fetched joke."}),
            fake_tool_result({"status": "success", "image_url": "https://example.com/dog.jpg"}),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=[
                "city_to_coords",
                "get_weather",
                "book_recs",
                "random_joke",
                "random_dog",
            ],
            model_name="demo-model",
        )

        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Plan a cozy Saturday in New York. Include the current weather, 2 book ideas about mystery, one joke, and a dog pic.",
        )

        self.assertTrue(result.used_fallback)
        self.assertIn("Weather:", result.answer)
        self.assertIn("Books:", result.answer)
        self.assertIn("Joke:", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 5)
        self.assertEqual(mock_react.call_count, 6)
        mock_reflection.assert_called_once()

    @patch("agent.orchestrator.llm_reflection_json", side_effect=ValueError("reflection boom"))
    @patch("agent.orchestrator.llm_react_json")
    async def test_reflection_failure_falls_back_to_grounded_draft(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "I should fetch a joke.", "action": "tool", "tool": "random_joke", "args": {}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is a joke for you."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [fake_tool_result({"joke": "A fetched joke."})]

        context = OrchestratorContext(history=[], tool_names=["random_joke"], model_name="demo-model")
        result = await orchestrate_interaction(tool_gateway=tool_gateway, context=context, user_prompt="Tell me a joke.")

        self.assertTrue(result.used_fallback)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 1)

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Here is your joke."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_post_joke_invalid_output_can_be_repaired_to_finish(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "I should fetch a joke.", "action": "tool", "tool": "random_joke", "args": {}},
            {"thought": "I have enough information.", "action": "finish", "final_answer": "Here is your joke."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [fake_tool_result({"joke": "A fetched joke."})]

        context = OrchestratorContext(history=[], tool_names=["random_joke"], model_name="demo-model")
        result = await orchestrate_interaction(tool_gateway=tool_gateway, context=context, user_prompt="Tell me a joke.")

        self.assertTrue(result.used_fallback)
        self.assertEqual(len(result.tool_observations), 1)
        self.assertEqual(result.answer, "Joke: A fetched joke.")
        self.assertEqual(tool_gateway.call_tool.await_count, 1)
        second_call_messages = mock_react.call_args_list[1].args[0]
        combined = "\n".join(message["content"] for message in second_call_messages)
        self.assertIn("Thought: I should fetch a joke.", combined)
        self.assertIn("Tool: random_joke", combined)
        self.assertIn("- Joke: A fetched joke.", combined)
        self.assertEqual(result.tool_observations[0].payload, '{"joke": "A fetched joke."}')

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Weather fetched."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_city_lookup_summary_is_available_to_follow_up_weather_step(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Need coordinates first.", "action": "tool", "tool": "city_to_coords", "args": {"city": "New York"}},
            {"thought": "Now get weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.71427, "longitude": -74.00597}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "city": "New York",
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "country": "United States",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "temperature": 6.1,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["city_to_coords", "get_weather"],
            model_name="demo-model",
        )
        await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="What's the weather in New York?",
        )

        second_call_messages = mock_react.call_args_list[1].args[0]
        combined = "\n".join(message["content"] for message in second_call_messages)
        self.assertIn("Tool: city_to_coords", combined)
        self.assertIn("- City Lookup: New York: 40.71427, -74.00597", combined)
        self.assertNotIn('"latitude": 40.71427', combined)
        self.assertNotIn('"longitude": -74.00597', combined)

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Here is the weather."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_prompt_coordinates_weather_with_empty_args_is_invalid(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Fetch weather.", "action": "tool", "tool": "get_weather", "args": {}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "temperature": 11.2,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="What's the weather for 41.85003, -87.65005?",
        )

        self.assertEqual(tool_gateway.call_tool.await_count, 0)
        self.assertIn("latitude and longitude are required", result.tool_observations[0].payload)
        self.assertEqual(
            result.answer,
            "Weekend Wizard Results\n- Weather: requested location unavailable (latitude and longitude are required)",
        )

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Here is the weather."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_duplicate_successful_get_weather_executes_only_once(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Need coordinates first.", "action": "tool", "tool": "city_to_coords", "args": {"city": "Chicago"}},
            {"thought": "Now get weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 41.85003, "longitude": -87.65005}},
            {"thought": "Get weather again.", "action": "tool", "tool": "get_weather", "args": {"latitude": 41.85003, "longitude": -87.65005}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "city": "Chicago",
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "country": "United States",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "temperature": 11.2,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["city_to_coords", "get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="What's the weather in Chicago?",
        )

        self.assertEqual(tool_gateway.call_tool.await_count, 2)
        self.assertEqual(len(result.tool_observations), 2)
        self.assertEqual(
            result.answer,
            "Weekend Wizard Results\n- City Lookup: Chicago: 41.85003, -87.65005\n- Weather: 41.85003, -87.65005: 11.2C, clear sky",
        )
        self.assertEqual(mock_react.call_count, 4)

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Here is the weather."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_explicit_equivalent_weather_args_count_as_duplicate(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Need coordinates first.", "action": "tool", "tool": "city_to_coords", "args": {"city": "Chicago"}},
            {"thought": "Now get weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 41.85003, "longitude": -87.65005}},
            {
                "thought": "Get weather again explicitly.",
                "action": "tool",
                "tool": "get_weather",
                "args": {"latitude": 41.85003, "longitude": -87.65005},
            },
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "city": "Chicago",
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "country": "United States",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "temperature": 11.2,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["city_to_coords", "get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="What's the weather in Chicago?",
        )

        self.assertEqual(tool_gateway.call_tool.await_count, 2)
        self.assertEqual(len(result.tool_observations), 2)
        self.assertEqual(
            result.answer,
            "Weekend Wizard Results\n- City Lookup: Chicago: 41.85003, -87.65005\n- Weather: 41.85003, -87.65005: 11.2C, clear sky",
        )
        self.assertEqual(mock_react.call_count, 4)

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Could not disambiguate weather."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_multiple_city_lookups_make_empty_arg_weather_invalid(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Need Chicago coordinates.", "action": "tool", "tool": "city_to_coords", "args": {"city": "Chicago"}},
            {"thought": "Need New York coordinates.", "action": "tool", "tool": "city_to_coords", "args": {"city": "New York"}},
            {"thought": "Now get weather.", "action": "tool", "tool": "get_weather", "args": {}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "city": "Chicago",
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "country": "United States",
                }
            ),
            fake_tool_result(
                {
                    "city": "New York",
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "country": "United States",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["city_to_coords", "get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="What's the weather in Chicago and New York?",
        )

        self.assertEqual(tool_gateway.call_tool.await_count, 2)
        self.assertEqual(len(result.tool_observations), 3)
        self.assertEqual(result.tool_observations[2].tool_name, "get_weather")
        self.assertIn("latitude and longitude", result.tool_observations[2].payload)

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Here are both weather results."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_using_their_coordinates_prompt_fetches_weather_for_both_cities_before_finish(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Resolve Chicago first.", "action": "tool", "tool": "city_to_coords", "args": {"city": "Chicago"}},
            {
                "thought": "Now fetch Chicago weather with explicit coordinates.",
                "action": "tool",
                "tool": "get_weather",
                "args": {"latitude": 41.85003, "longitude": -87.65005},
            },
            {"thought": "Resolve New York next.", "action": "tool", "tool": "city_to_coords", "args": {"city": "New York"}},
            {
                "thought": "Now fetch New York weather with explicit coordinates.",
                "action": "tool",
                "tool": "get_weather",
                "args": {"latitude": 40.71427, "longitude": -74.00597},
            },
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here are both weather results."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "city": "Chicago",
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "country": "United States",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "temperature": 11.2,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
            fake_tool_result(
                {
                    "city": "New York",
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "country": "United States",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "temperature": 6.1,
                    "temperature_unit": "C",
                    "weather_summary": "light rain",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["city_to_coords", "get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Get the weather for Chicago and New York using their coordinates.",
        )

        self.assertEqual(tool_gateway.call_tool.await_count, 4)
        weather_args = [
            observation.args
            for observation in result.tool_observations
            if observation.tool_name == "get_weather"
        ]
        self.assertEqual(
            weather_args,
            [
                {"latitude": 41.85003, "longitude": -87.65005},
                {"latitude": 40.71427, "longitude": -74.00597},
            ],
        )
        self.assertEqual(
            result.answer,
            "Weekend Wizard Results\n- City Lookup: Chicago: 41.85003, -87.65005\n- Weather: 41.85003, -87.65005: 11.2C, clear sky\n- City Lookup: New York: 40.71427, -74.00597\n- Weather: 40.71427, -74.00597: 6.1C, light rain",
        )

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Here are both weather results."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_duplicate_chicago_weather_is_skipped_and_loop_continues_to_new_york(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Resolve Chicago first.", "action": "tool", "tool": "city_to_coords", "args": {"city": "Chicago"}},
            {"thought": "Fetch Chicago weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 41.85003, "longitude": -87.65005}},
            {"thought": "Fetch Chicago weather again.", "action": "tool", "tool": "get_weather", "args": {"latitude": 41.85003, "longitude": -87.65005}},
            {"thought": "Resolve New York now.", "action": "tool", "tool": "city_to_coords", "args": {"city": "New York"}},
            {"thought": "Fetch New York weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.71427, "longitude": -74.00597}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here are both weather results."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "city": "Chicago",
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "country": "United States",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "temperature": 11.2,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
            fake_tool_result(
                {
                    "city": "New York",
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "country": "United States",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "temperature": 6.1,
                    "temperature_unit": "C",
                    "weather_summary": "light rain",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["city_to_coords", "get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Get the weather for Chicago and New York using their coordinates.",
        )

        self.assertEqual(tool_gateway.call_tool.await_count, 4)
        self.assertEqual(mock_react.call_count, 6)
        weather_args = [
            observation.args
            for observation in result.tool_observations
            if observation.tool_name == "get_weather"
        ]
        self.assertEqual(
            weather_args,
            [
                {"latitude": 41.85003, "longitude": -87.65005},
                {"latitude": 40.71427, "longitude": -74.00597},
            ],
        )
        self.assertEqual(
            result.answer,
            "Weekend Wizard Results\n- City Lookup: Chicago: 41.85003, -87.65005\n- Weather: 41.85003, -87.65005: 11.2C, clear sky\n- City Lookup: New York: 40.71427, -74.00597\n- Weather: 40.71427, -74.00597: 6.1C, light rain",
        )

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Here are both weather results."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_weather_no_longer_blocks_early_finish_with_custom_progress_gate(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Resolve Chicago first.", "action": "tool", "tool": "city_to_coords", "args": {"city": "Chicago"}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "city": "Chicago",
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "country": "United States",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["city_to_coords", "get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="What's the weather in Chicago?",
        )

        self.assertEqual(tool_gateway.call_tool.await_count, 1)
        self.assertEqual(mock_react.call_count, 2)
        self.assertEqual(
            result.answer,
            "Weekend Wizard Results\n- City Lookup: Chicago: 41.85003, -87.65005",
        )

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Book ideas fetched."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_book_recs_follow_up_planning_sees_compact_summary_context(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Need books first.", "action": "tool", "tool": "book_recs", "args": {"topic": "mystery", "limit": 2}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here are your book ideas."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "topic": "mystery",
                    "count": 2,
                    "results": [
                        {"title": "A Caribbean Mystery", "author": "Agatha Christie"},
                        {"title": "The Mysterious Affair at Styles", "author": "Agatha Christie"},
                    ],
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["book_recs"],
            model_name="demo-model",
        )
        await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Give me 2 mystery book ideas.",
        )

        second_call_messages = mock_react.call_args_list[1].args[0]
        combined = "\n".join(message["content"] for message in second_call_messages)
        self.assertIn("Tool: book_recs", combined)
        self.assertIn("- Books: A Caribbean Mystery by Agatha Christie; The Mysterious Affair at Styles by Agatha Christie", combined)

    @patch(
        "agent.orchestrator.llm_reflection_json",
        return_value={"answer": "Weather failed, but here's a joke."},
    )
    @patch("agent.orchestrator.llm_react_json")
    async def test_failed_tool_summary_includes_safe_detail_for_next_react_step(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Fetch weather first.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.7128, "longitude": -74.0060}},
            {"thought": "Now fetch a joke.", "action": "tool", "tool": "random_joke", "args": {}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the latest weather and a joke."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            ToolInvocationError("GET https://internal.example.local/weather?token=secret timed out"),
            fake_tool_result({"joke": "A fetched joke."}),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["get_weather", "random_joke"],
            model_name="demo-model",
        )
        await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Give me the weather and a joke for 40.7128, -74.0060.",
        )

        second_call_messages = mock_react.call_args_list[1].args[0]
        combined = "\n".join(message["content"] for message in second_call_messages)
        self.assertIn("Tool: get_weather", combined)
        self.assertIn("- Weather: 40.7128, -74.006 unavailable (tool execution failed)", combined)
        self.assertNotIn("internal.example.local", combined)
        self.assertNotIn("token=secret", combined)
        self.assertNotIn('{"error": "get_weather failed"', combined)

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Weather retried."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_failed_duplicate_call_remains_retryable(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Fetch weather first.", "action": "tool", "tool": "get_weather", "args": {"latitude": 41.85003, "longitude": -87.65005}},
            {"thought": "Retry weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 41.85003, "longitude": -87.65005}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            ToolInvocationError("timeout"),
            fake_tool_result(
                {
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "temperature": 11.2,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="What's the weather for 41.85003, -87.65005?",
        )

        self.assertEqual(tool_gateway.call_tool.await_count, 2)
        self.assertEqual(len(result.tool_observations), 2)
        self.assertEqual(
            result.answer,
            "Weekend Wizard Results\n- Weather: 41.85003, -87.65005 unavailable (tool execution failed)\n- Weather: 41.85003, -87.65005: 11.2C, clear sky",
        )

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Here are both weather results."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_different_weather_args_are_not_blocked(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Fetch first weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 41.85003, "longitude": -87.65005}},
            {"thought": "Fetch second weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.71427, "longitude": -74.00597}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "temperature": 11.2,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "temperature": 6.1,
                    "temperature_unit": "C",
                    "weather_summary": "light rain",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Compare the weather for Chicago and New York.",
        )

        self.assertEqual(tool_gateway.call_tool.await_count, 2)
        self.assertEqual(len(result.tool_observations), 2)
        self.assertEqual(
            result.answer,
            "Weekend Wizard Results\n- Weather: 41.85003, -87.65005: 11.2C, clear sky\n- Weather: 40.71427, -74.00597: 6.1C, light rain",
        )

    @patch("agent.orchestrator.llm_react_json")
    async def test_invalid_decision_returns_failure_message(self, mock_react: Mock) -> None:
        mock_react.return_value = {
            "thought": "I should use a tool.",
            "action": "tool",
            "args": {},
        }

        tool_gateway = AsyncMock()
        context = OrchestratorContext(
            history=[],
            tool_names=["get_weather", "city_to_coords"],
            model_name="demo-model",
        )

        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="What's the weather in New York?",
        )

        self.assertFalse(result.used_fallback)
        self.assertEqual(result.tool_observations, [])
        self.assertIn("couldn't complete a reliable weekend wizard turn", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 0)

    @patch("agent.orchestrator.llm_react_json")
    async def test_planning_failure_after_tool_success_preserves_observations(
        self,
        mock_react: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Fetch a joke first.", "action": "tool", "tool": "random_joke", "args": {}},
            {"thought": "I should use a tool.", "action": "tool", "args": {}},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [fake_tool_result({"joke": "A fetched joke."})]

        context = OrchestratorContext(history=[], tool_names=["random_joke"], model_name="demo-model")
        result = await orchestrate_interaction(tool_gateway=tool_gateway, context=context, user_prompt="Tell me a joke.")

        self.assertTrue(result.used_fallback)
        self.assertEqual(len(result.tool_observations), 1)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 1)

    @patch(
        "agent.orchestrator.llm_reflection_json",
        return_value={
            "answer": (
                "Weekend Wizard Plan\n"
                "- Weather: unavailable (weather request failed)\n"
                "- Joke: A fetched joke."
            )
        },
    )
    @patch("agent.orchestrator.llm_react_json")
    async def test_tool_failures_are_recorded_and_remaining_steps_continue(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Fetch weather first.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.7128, "longitude": -74.0060}},
            {"thought": "Now fetch a joke.", "action": "tool", "tool": "random_joke", "args": {}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the latest weather and a joke."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            ToolInvocationError("GET https://internal.example.local/weather?token=secret timed out"),
            fake_tool_result({"joke": "A fetched joke."}),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["get_weather", "random_joke"],
            model_name="demo-model",
        )

        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Give me the weather and a joke for 40.7128, -74.0060.",
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(len(result.tool_observations), 2)
        self.assertEqual(
            result.answer,
            "Weekend Wizard Results\n- Weather: 40.7128, -74.006 unavailable (tool execution failed)\n- Joke: A fetched joke.",
        )
        self.assertNotIn("internal.example.local", result.answer)
        self.assertNotIn("token=secret", result.answer)
        self.assertEqual(
            result.tool_observations[0].payload,
            '{"error": "get_weather failed", "details": "' + SAFE_TOOL_INVOCATION_DETAIL + '"}',
        )
        self.assertNotIn("internal.example.local", result.tool_observations[0].payload)
        self.assertEqual(tool_gateway.call_tool.await_count, 2)

    @patch(
        "agent.orchestrator.llm_reflection_json",
        return_value={"answer": "A fetched joke: hope that brightens your day."},
    )
    @patch("agent.orchestrator.llm_react_json")
    async def test_reflection_can_polish_grounded_answer_without_dropping_core_fact(
        self,
        mock_react: Mock,
        mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "I should fetch a joke.", "action": "tool", "tool": "random_joke", "args": {}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is a joke for you."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [fake_tool_result({"joke": "A fetched joke."})]

        context = OrchestratorContext(history=[], tool_names=["random_joke"], model_name="demo-model")
        result = await orchestrate_interaction(tool_gateway=tool_gateway, context=context, user_prompt="Tell me a joke.")

        mock_reflection.assert_called_once()
        self.assertFalse(result.used_fallback)
        self.assertIn("A fetched joke.", result.answer)
        self.assertIn("brightens your day", result.answer)

    @patch(
        "agent.orchestrator.llm_reflection_json",
        return_value={"answer": "The weather in Paris is 18.3°C and mainly clear right now."},
    )
    @patch("agent.orchestrator.llm_react_json")
    async def test_reflection_can_return_natural_weather_answer_without_city_lookup_scaffolding(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Need coordinates first.", "action": "tool", "tool": "city_to_coords", "args": {"city": "Paris"}},
            {"thought": "Now get weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 48.85341, "longitude": 2.3488}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "city": "Paris",
                    "latitude": 48.85341,
                    "longitude": 2.3488,
                    "country": "France",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 48.85341,
                    "longitude": 2.3488,
                    "temperature": 18.3,
                    "temperature_unit": "°C",
                    "weather_summary": "mainly clear",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["city_to_coords", "get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="what is weather in paris",
        )

        self.assertFalse(result.used_fallback)
        self.assertIn("18.3°C", result.answer)
        self.assertIn("mainly clear", result.answer)
        self.assertNotIn("City Lookup", result.answer)

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Here is a joke for you."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_grounded_draft_wins_when_reflection_drops_fetched_fact(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "I should fetch a joke.", "action": "tool", "tool": "random_joke", "args": {}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is a joke for you."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [fake_tool_result({"joke": "A fetched joke."})]

        context = OrchestratorContext(history=[], tool_names=["random_joke"], model_name="demo-model")
        result = await orchestrate_interaction(tool_gateway=tool_gateway, context=context, user_prompt="Tell me a joke.")

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.answer, "Joke: A fetched joke.")

    @patch(
        "agent.orchestrator.llm_reflection_json",
        return_value={"answer": "Weekend Wizard Plan\n- Joke: A fetched joke."},
    )
    @patch("agent.orchestrator.llm_react_json")
    async def test_grounded_draft_wins_when_reflection_hides_failure(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Fetch weather first.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.7128, "longitude": -74.0060}},
            {"thought": "Now fetch a joke.", "action": "tool", "tool": "random_joke", "args": {}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the latest weather and a joke."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            ToolInvocationError("GET https://internal.example.local/weather?token=secret timed out"),
            fake_tool_result({"joke": "A fetched joke."}),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["get_weather", "random_joke"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Give me the weather and a joke for 40.7128, -74.0060.",
        )

        self.assertTrue(result.used_fallback)
        self.assertIn("unavailable", result.answer)
        self.assertIn("A fetched joke.", result.answer)

    @patch(
        "agent.orchestrator.llm_reflection_json",
        return_value={"answer": "Here is the weather comparison."},
    )
    @patch("agent.orchestrator.llm_react_json")
    async def test_grounded_draft_wins_when_reflection_collapses_multiple_results(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Fetch first weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 41.85003, "longitude": -87.65005}},
            {"thought": "Fetch second weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.71427, "longitude": -74.00597}},
            {"thought": "I can answer now.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "latitude": 41.85003,
                    "longitude": -87.65005,
                    "temperature": 11.2,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
            fake_tool_result(
                {
                    "latitude": 40.71427,
                    "longitude": -74.00597,
                    "temperature": 6.1,
                    "temperature_unit": "C",
                    "weather_summary": "light rain",
                }
            ),
        ]

        context = OrchestratorContext(
            history=[],
            tool_names=["get_weather"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Compare the weather for Chicago and New York.",
        )

        self.assertTrue(result.used_fallback)
        self.assertIn("41.85003, -87.65005", result.answer)
        self.assertIn("40.71427, -74.00597", result.answer)

    @patch("agent.orchestrator.llm_reflection_json", return_value={"answer": "Here is the weather."})
    @patch("agent.orchestrator.llm_react_json")
    async def test_repeated_duplicate_successful_call_finalizes_early(
        self,
        mock_react: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Fetch weather first.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.7128, "longitude": -74.0060}},
            {"thought": "Fetch weather again.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.7128, "longitude": -74.0060}},
            {"thought": "Still fetch weather.", "action": "tool", "tool": "get_weather", "args": {"latitude": 40.7128, "longitude": -74.0060}},
            {"thought": "This should never be reached.", "action": "finish", "final_answer": "Here is the weather."},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "temperature": 29.1,
                    "temperature_unit": "C",
                    "weather_summary": "clear sky",
                }
            ),
        ]

        context = OrchestratorContext(history=[], tool_names=["get_weather"], model_name="demo-model")
        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="What is the weather in New York?",
        )

        self.assertTrue(result.used_fallback)
        self.assertEqual(tool_gateway.call_tool.await_count, 1)
        self.assertEqual(len(result.tool_observations), 1)
        self.assertEqual(mock_react.call_count, 3)
        self.assertIn("29.1C", result.answer)

    def test_validate_react_decision_semantics_rejects_unknown_tool(self) -> None:
        decision = {
            "thought": "I should use a tool.",
            "action": "tool",
            "tool": "not_a_tool",
            "args": {},
        }

        with self.assertRaisesRegex(ValueError, "unsupported tool"):
            validate_react_decision_semantics(
                decision=validate_react_decision(decision),
                available_tools=["city_to_coords", "get_weather", "book_recs"],
            )

    def test_validate_react_decision_rejects_finish_without_answer(self) -> None:
        decision = {
            "thought": "I am done.",
            "action": "finish",
        }

        with self.assertRaisesRegex(ValueError, "final_answer"):
            validate_react_decision(decision)


if __name__ == "__main__":
    unittest.main()
