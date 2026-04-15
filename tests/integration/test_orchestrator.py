from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent.orchestrator import orchestrate_interaction
from config.config import get_settings
from schemas.agent import OrchestratorContext


def fake_tool_result(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])


class OrchestratorIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @patch("agent.orchestrator.llm_json")
    async def test_city_prompt_flows_to_grounded_final_answer(
        self,
        mock_llm_json: Mock,
    ) -> None:
        mock_llm_json.side_effect = [
            {"action": "city_to_coords", "args": {"city": "New York"}},
            {"action": "get_weather", "args": {"latitude": 40.71427, "longitude": -74.00597}},
            {"action": "book_recs", "args": {"topic": "mystery", "limit": 3}},
            {"action": "random_joke", "args": {}},
            {"action": "random_dog", "args": {}},
            {"action": "final", "answer": "placeholder"},
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
                    "temperature_unit": "°C",
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
            history=[{"role": "system", "content": "system prompt"}],
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

        final_answer = result.answer
        self.assertFalse(result.used_step_limit_fallback)

        self.assertEqual(context.history[-1]["role"], "assistant")
        self.assertIn("Weekend Wizard Plan", final_answer)
        self.assertIn("6.1°C, clear sky", final_answer)
        self.assertIn("A Caribbean Mystery by Agatha Christie", final_answer)
        self.assertIn("A fetched joke.", final_answer)
        self.assertIn("https://example.com/dog.jpg", final_answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 5)
        self.assertEqual(len(result.tool_observations), 5)

    @patch("agent.orchestrator.llm_json")
    async def test_final_is_reprompted_instead_of_using_stub_policy_in_normal_mode(
        self,
        mock_llm_json: Mock,
    ) -> None:
        mock_llm_json.side_effect = [
            {"action": "final", "answer": "too early"},
            {"action": "random_joke", "args": {}},
            {"action": "final", "answer": "placeholder"},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result({"joke": "A fetched joke."}),
        ]

        context = OrchestratorContext(
            history=[{"role": "system", "content": "system prompt"}],
            tool_names=["random_joke"],
            model_name="demo-model",
        )

        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Tell me a joke.",
        )

        self.assertFalse(result.used_step_limit_fallback)
        self.assertEqual(tool_gateway.call_tool.await_count, 1)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(mock_llm_json.call_count, 3)
        self.assertEqual(context.history[-1]["role"], "assistant")

    @patch("agent.orchestrator.llm_json")
    async def test_reused_context_does_not_treat_old_tool_results_as_current_turn_results(
        self,
        mock_llm_json: Mock,
    ) -> None:
        mock_llm_json.side_effect = [
            {"action": "random_joke", "args": {}},
            {"action": "final", "answer": "first"},
            {"action": "final", "answer": "too early"},
            {"action": "random_joke", "args": {}},
            {"action": "final", "answer": "second"},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result({"joke": "First fetched joke."}),
            fake_tool_result({"joke": "Second fetched joke."}),
        ]

        context = OrchestratorContext(
            history=[{"role": "system", "content": "system prompt"}],
            tool_names=["random_joke"],
            model_name="demo-model",
        )

        first_result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Tell me a joke.",
        )
        second_result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Tell me another joke.",
        )

        self.assertFalse(first_result.used_step_limit_fallback)
        self.assertFalse(second_result.used_step_limit_fallback)
        self.assertIn("First fetched joke.", first_result.answer)
        self.assertIn("Second fetched joke.", second_result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 2)
        self.assertEqual(mock_llm_json.call_count, 5)

    @patch("agent.orchestrator.llm_json")
    async def test_step_limit_fallback_returns_grounded_answer_after_required_tools(
        self,
        mock_llm_json: Mock,
    ) -> None:
        get_settings.cache_clear()
        mock_llm_json.side_effect = [
            {"action": "get_weather", "args": {"latitude": 40.7128, "longitude": -74.0060}},
            {"action": "book_recs", "args": {"topic": "mystery", "limit": 3}},
            {"action": "random_joke", "args": {}},
            {"action": "random_dog", "args": {}},
            {"action": "random_joke", "args": {}},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result(
                {
                    "latitude": 40.7128,
                    "longitude": -74.0060,
                    "temperature": 4.4,
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
            fake_tool_result({"joke": "A second fetched joke."}),
        ]

        context = OrchestratorContext(
            history=[{"role": "system", "content": "system prompt"}],
            tool_names=[
                "get_weather",
                "book_recs",
                "random_joke",
                "random_dog",
            ],
            model_name="demo-model",
        )

        try:
            with patch.dict(os.environ, {"WEEKEND_WIZARD_MAX_STEPS": "5"}, clear=False):
                get_settings.cache_clear()
                result = await orchestrate_interaction(
                    tool_gateway=tool_gateway,
                    context=context,
                    user_prompt="Plan a cozy Saturday in New York at (40.7128, -74.0060). Include the current weather, 2 book ideas about mystery, one joke, and a dog pic.",
                )
        finally:
            get_settings.cache_clear()

        self.assertTrue(result.used_step_limit_fallback)
        self.assertEqual(context.history[-1]["role"], "assistant")
        self.assertIn("Weekend Wizard Plan", result.answer)
        self.assertIn("4.4C, clear sky", result.answer)
        self.assertIn("A Caribbean Mystery by Agatha Christie", result.answer)
        self.assertIn("A second fetched joke.", result.answer)
        self.assertIn("https://example.com/dog.jpg", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 5)


if __name__ == "__main__":
    unittest.main()
