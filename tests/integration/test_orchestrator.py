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
    async def test_city_prompt_flows_to_grounded_final_answer(self, mock_llm_json: Mock) -> None:
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
        mock_llm_json.assert_not_called()

        self.assertEqual(context.history[-1]["role"], "assistant")
        self.assertIn("Weekend Wizard Plan", final_answer)
        self.assertIn("6.1C, clear sky", final_answer)
        self.assertIn("A Caribbean Mystery by Agatha Christie", final_answer)
        self.assertIn("A fetched joke.", final_answer)
        self.assertIn("https://example.com/dog.jpg", final_answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 5)
        self.assertEqual(len(result.tool_observations), 5)

    @patch("agent.orchestrator.llm_json")
    async def test_single_joke_prompt_uses_deterministic_flow(self, mock_llm_json: Mock) -> None:
        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [fake_tool_result({"joke": "A fetched joke."})]

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
        mock_llm_json.assert_not_called()
        self.assertEqual(context.history[-1]["role"], "assistant")

    @patch("agent.orchestrator.llm_json")
    async def test_reused_context_does_not_treat_old_tool_results_as_current_turn_results(
        self,
        mock_llm_json: Mock,
    ) -> None:
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
        mock_llm_json.assert_not_called()

    @patch("agent.orchestrator.llm_json")
    async def test_open_ended_prompt_can_still_use_model_led_fallback_controller(
        self,
        mock_llm_json: Mock,
    ) -> None:
        mock_llm_json.side_effect = [
            {"action": "random_joke", "args": {}},
            {"action": "final", "answer": "placeholder"},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [fake_tool_result({"joke": "A fetched joke."})]

        context = OrchestratorContext(
            history=[{"role": "system", "content": "system prompt"}],
            tool_names=["random_joke"],
            model_name="demo-model",
        )

        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Surprise me with something fun for tonight.",
        )

        self.assertFalse(result.used_step_limit_fallback)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 1)
        self.assertEqual(mock_llm_json.call_count, 2)

    @patch("agent.orchestrator.llm_json")
    async def test_step_limit_fallback_still_works_for_non_deterministic_controller_path(
        self,
        mock_llm_json: Mock,
    ) -> None:
        get_settings.cache_clear()
        mock_llm_json.side_effect = [
            {"action": "random_joke", "args": {}},
            {"action": "random_joke", "args": {}},
        ]

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [
            fake_tool_result({"joke": "A fetched joke."}),
            fake_tool_result({"joke": "A second fetched joke."}),
        ]

        context = OrchestratorContext(
            history=[{"role": "system", "content": "system prompt"}],
            tool_names=["random_joke"],
            model_name="demo-model",
        )

        try:
            with patch.dict(os.environ, {"WEEKEND_WIZARD_MAX_STEPS": "2"}, clear=False):
                get_settings.cache_clear()
                result = await orchestrate_interaction(
                    tool_gateway=tool_gateway,
                    context=context,
                    user_prompt="Keep riffing until you hit the limit.",
                )
        finally:
            get_settings.cache_clear()

        self.assertTrue(result.used_step_limit_fallback)
        self.assertIn("A second fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 2)


if __name__ == "__main__":
    unittest.main()
