from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent.orchestrator import orchestrate_interaction, validate_react_decision_semantics
from mcp_runtime.client import ToolInvocationError
from schemas.agent import OrchestratorContext, validate_react_decision


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
    @patch("agent.orchestrator.llm_react_json")
    async def test_city_prompt_flows_to_reflected_grounded_final_answer(
        self,
        mock_react: Mock,
        mock_reflection: Mock,
    ) -> None:
        mock_react.side_effect = [
            {"thought": "Need coordinates first.", "action": "tool", "tool": "city_to_coords", "args": {"city": "New York"}},
            {"thought": "Now get weather.", "action": "tool", "tool": "get_weather", "args": {}},
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

        self.assertFalse(result.used_fallback)
        self.assertIn("Weekend Wizard Plan", result.answer)
        self.assertIn("A fetched joke.", result.answer)
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

        self.assertFalse(result.used_fallback)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 1)

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

        self.assertFalse(result.used_fallback)
        self.assertEqual(len(result.tool_observations), 2)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 2)

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
