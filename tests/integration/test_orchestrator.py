from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from agent.orchestrator import orchestrate_interaction
from guardrails.plans import validate_plan_semantics
from mcp_runtime.client import ToolInvocationError
from schemas.agent import OrchestratorContext, ReflectionResult, validate_execution_plan


def fake_tool_result(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])


class OrchestratorIntegrationTests(unittest.IsolatedAsyncioTestCase):
    @patch(
        "agent.orchestrator.llm_reflection_json",
        return_value=ReflectionResult(
            answer=(
                "Here is your cozy Saturday plan: expect 6.1C and clear skies, "
                "read A Caribbean Mystery and The Mysterious Affair at Styles, "
                "enjoy this joke: A fetched joke., and check this dog photo: https://example.com/dog.jpg"
            )
        ),
    )
    @patch("agent.orchestrator.llm_plan_json")
    async def test_city_prompt_returns_reflection_as_final_answer(
        self,
        mock_plan: Mock,
        mock_reflection: Mock,
    ) -> None:
        mock_plan.return_value = validate_execution_plan(
            {
                "goal": "weekend_plan",
                "location": {"city": "New York"},
                "execution_steps": [
                    {"tool": "city_to_coords", "args": {"city": "New York"}},
                    {"tool": "get_weather", "args": {}},
                    {"tool": "book_recs", "args": {"topic": "mystery", "limit": 2}},
                    {"tool": "random_joke", "args": {}},
                    {"tool": "random_dog", "args": {}},
                ],
            }
        )

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
        self.assertEqual(
            result.answer,
            "Here is your cozy Saturday plan: expect 6.1C and clear skies, read A Caribbean Mystery and The Mysterious Affair at Styles, enjoy this joke: A fetched joke., and check this dog photo: https://example.com/dog.jpg",
        )
        self.assertNotIn("Weekend Wizard Plan\n- Weather:", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 5)
        mock_reflection.assert_called_once()

    @patch("agent.orchestrator.llm_reflection_json", side_effect=ValueError("reflection boom"))
    @patch("agent.orchestrator.llm_plan_json")
    async def test_reflection_failure_falls_back_to_grounded_draft(
        self,
        mock_plan: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_plan.return_value = validate_execution_plan(
            {
                "goal": "joke",
                "execution_steps": [{"tool": "random_joke", "args": {}}],
            }
        )

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [fake_tool_result({"joke": "A fetched joke."})]

        context = OrchestratorContext(
            history=[],
            tool_names=["random_joke"],
            model_name="demo-model",
        )
        result = await orchestrate_interaction(tool_gateway=tool_gateway, context=context, user_prompt="Tell me a joke.")

        self.assertFalse(result.used_fallback)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 1)

    @patch("agent.orchestrator.llm_plan_json")
    async def test_invalid_plan_returns_planner_failure_message(self, mock_plan: Mock) -> None:
        mock_plan.return_value = validate_execution_plan(
            {
                "goal": "weekend_plan",
                "execution_steps": [{"tool": "get_weather", "args": {}}],
            }
        )

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

        self.assertTrue(result.used_fallback)
        self.assertEqual(result.tool_observations, [])
        self.assertIn("couldn't build a reliable weekend plan", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 0)

    @patch(
        "agent.orchestrator.llm_reflection_json",
        return_value=ReflectionResult(answer="A fetched joke."),
    )
    @patch("agent.orchestrator.validate_execution_plan", create=True)
    @patch("agent.orchestrator.llm_plan_json")
    async def test_orchestrator_does_not_revalidate_typed_plan_schema(
        self,
        mock_plan: Mock,
        mock_validate_execution_plan: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_plan.return_value = validate_execution_plan(
            {
                "goal": "joke",
                "execution_steps": [{"tool": "random_joke", "args": {}}],
            }
        )

        tool_gateway = AsyncMock()
        tool_gateway.call_tool.side_effect = [fake_tool_result({"joke": "A fetched joke."})]

        context = OrchestratorContext(
            history=[],
            tool_names=["random_joke"],
            model_name="demo-model",
        )

        result = await orchestrate_interaction(
            tool_gateway=tool_gateway,
            context=context,
            user_prompt="Tell me a joke.",
        )

        mock_validate_execution_plan.assert_not_called()
        self.assertFalse(result.used_fallback)
        self.assertEqual(tool_gateway.call_tool.await_count, 1)

    @patch(
        "agent.orchestrator.llm_reflection_json",
        return_value=ReflectionResult(
            answer=(
                "Weekend Wizard Plan\n"
                "- Weather: unavailable (weather request failed)\n"
                "- Joke: A fetched joke."
            )
        ),
    )
    @patch("agent.orchestrator.llm_plan_json")
    async def test_tool_failures_are_recorded_and_remaining_steps_continue(
        self,
        mock_plan: Mock,
        _mock_reflection: Mock,
    ) -> None:
        mock_plan.return_value = validate_execution_plan(
            {
                "goal": "weekend_plan",
                "location": {"latitude": 40.7128, "longitude": -74.0060},
                "execution_steps": [
                    {"tool": "get_weather", "args": {"latitude": 40.7128, "longitude": -74.0060}},
                    {"tool": "random_joke", "args": {}},
                ],
            }
        )

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

        self.assertTrue(result.used_fallback)
        self.assertEqual(len(result.tool_observations), 2)
        self.assertIn("A fetched joke.", result.answer)
        self.assertEqual(tool_gateway.call_tool.await_count, 2)

    def test_validate_plan_semantics_rejects_unrequested_optional_tool(self) -> None:
        plan = {
            "goal": "weekend_plan",
            "location": {"city": "New York"},
            "execution_steps": [
                {"tool": "city_to_coords", "args": {"city": "New York"}},
                {"tool": "get_weather", "args": {}},
                {"tool": "book_recs", "args": {"topic": "mystery", "limit": 3}},
                {"tool": "random_joke", "args": {}},
                {"tool": "random_dog", "args": {}},
                {"tool": "trivia", "args": {}},
            ],
        }

        with self.assertRaisesRegex(ValueError, "unrequested tools"):
            validate_plan_semantics(
                plan=validate_execution_plan(plan),
                available_tools=["city_to_coords", "get_weather", "book_recs", "random_joke", "random_dog", "trivia"],
                user_prompt="Plan a cozy Saturday in New York with today's weather, 3 mystery book ideas, a joke, and a dog pic.",
            )

    def test_validate_plan_semantics_rejects_city_lookup_when_coords_already_exist(self) -> None:
        plan = {
            "goal": "weekend_plan",
            "location": {"city": "New York", "latitude": 40.7128, "longitude": -74.0060},
            "execution_steps": [
                {"tool": "city_to_coords", "args": {"city": "New York"}},
                {"tool": "get_weather", "args": {"latitude": 40.7128, "longitude": -74.0060}},
                {"tool": "book_recs", "args": {"topic": "mystery", "limit": 3}},
            ],
        }

        with self.assertRaisesRegex(ValueError, "coordinates were already provided"):
            validate_plan_semantics(
                plan=validate_execution_plan(plan),
                available_tools=["city_to_coords", "get_weather", "book_recs"],
                user_prompt="Plan a cozy Saturday in New York at (40.7128, -74.0060) with weather and 3 mystery book ideas.",
            )

    def test_validate_plan_semantics_rejects_bonus_joke_for_books_only_weekend_prompt(self) -> None:
        plan = {
            "goal": "weekend_plan",
            "location": {"city": "Las Vegas"},
            "execution_steps": [
                {"tool": "book_recs", "args": {"topic": "adventure", "limit": 3}},
                {"tool": "random_joke", "args": {}},
            ],
        }

        with self.assertRaisesRegex(ValueError, "unrequested tools"):
            validate_plan_semantics(
                plan=validate_execution_plan(plan),
                available_tools=["book_recs", "random_joke"],
                user_prompt="Plan a weekend in Las Vegas with 3 adventure books.",
            )

if __name__ == "__main__":
    unittest.main()
