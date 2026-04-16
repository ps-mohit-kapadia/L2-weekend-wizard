from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent.orchestrator import orchestrate_interaction
from mcp_runtime.client import ToolInvocationError
from schemas.agent import OrchestratorContext


def fake_tool_result(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])


class OrchestratorIntegrationTests(unittest.IsolatedAsyncioTestCase):
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
    @patch("agent.orchestrator.llm_plan_json")
    async def test_city_prompt_flows_to_reflected_grounded_final_answer(
        self,
        mock_plan: Mock,
        mock_reflection: Mock,
    ) -> None:
        mock_plan.return_value = {
            "goal": "weekend_plan",
            "location": {"city": "New York"},
            "book_topic": "mystery",
            "requested_tools": ["get_weather", "book_recs", "random_joke", "random_dog"],
            "execution_steps": [
                {"tool": "city_to_coords", "args": {"city": "New York"}},
                {"tool": "get_weather", "args": {}},
                {"tool": "book_recs", "args": {"param": "mystery", "limit": 2}},
                {"tool": "random_joke", "args": {}},
                {"tool": "random_dog", "args": {}},
            ],
        }

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

        self.assertFalse(result.used_step_limit_fallback)
        self.assertIn("Weekend Wizard Plan", result.answer)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 5)
        mock_reflection.assert_called_once()

    @patch("agent.orchestrator.llm_reflection_json", side_effect=ValueError("reflection boom"))
    @patch("agent.orchestrator.llm_plan_json")
    async def test_reflection_failure_falls_back_to_grounded_draft(
        self,
        mock_plan: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_plan.return_value = {
            "goal": "joke",
            "requested_tools": ["random_joke"],
            "execution_steps": [{"tool": "random_joke", "args": {}}],
        }

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [fake_tool_result({"joke": "A fetched joke."})]

        context = OrchestratorContext(history=[], tool_names=["random_joke"], model_name="demo-model")
        result = await orchestrate_interaction(tool_gateway=tool_gateway, context=context, user_prompt="Tell me a joke.")

        self.assertFalse(result.used_step_limit_fallback)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 1)

    @patch("agent.orchestrator.llm_plan_json")
    async def test_invalid_plan_returns_planner_failure_message(self, mock_plan: Mock) -> None:
        mock_plan.return_value = {
            "goal": "weekend_plan",
            "requested_tools": ["get_weather"],
            "execution_steps": [{"tool": "get_weather", "args": {}}],
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

        self.assertFalse(result.used_step_limit_fallback)
        self.assertEqual(result.tool_observations, [])
        self.assertIn("couldn't build a reliable weekend plan", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 0)

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
    @patch("agent.orchestrator.llm_plan_json")
    async def test_tool_failures_are_recorded_and_remaining_steps_continue(
        self,
        mock_plan: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_plan.return_value = {
            "goal": "weekend_plan",
            "location": {"latitude": 40.7128, "longitude": -74.0060},
            "requested_tools": ["get_weather", "random_joke"],
            "execution_steps": [
                {"tool": "get_weather", "args": {"latitude": 40.7128, "longitude": -74.0060}},
                {"tool": "random_joke", "args": {}},
            ],
        }

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            ToolInvocationError("weather request failed"),
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

        self.assertFalse(result.used_step_limit_fallback)
        self.assertEqual(len(result.tool_observations), 2)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 2)


if __name__ == "__main__":
    unittest.main()
