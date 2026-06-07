from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from claude_sdk_agent.config import get_default_config
from claude_sdk_agent.prompts import build_system_prompt
from claude_sdk_agent.runner import build_agent_options, run_claude_sdk_prompt
from claude_sdk_agent.smoke import get_smoke_scenarios, run_smoke_scenarios
from claude_sdk_agent.tools import (
    book_recs_tool,
    build_sdk_server,
    city_to_coords_tool,
    get_weather_tool,
    random_dog_tool,
    random_joke_tool,
    trivia_tool,
)
from schemas.tools import JokeResult


class ClaudeSdkAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_get_smoke_scenarios_returns_two_representative_cases(self) -> None:
        scenarios = get_smoke_scenarios()

        self.assertEqual(len(scenarios), 2)
        self.assertEqual(scenarios[0].label, "joke-only")
        self.assertEqual(scenarios[0].prompt, "Tell me a joke.")
        self.assertEqual(scenarios[1].label, "weekend-multi-tool")
        self.assertIn("New York", scenarios[1].prompt)
        self.assertIn("mystery books", scenarios[1].prompt)

    def test_build_system_prompt_covers_minimality_and_key_tool_rules(self) -> None:
        prompt = build_system_prompt()

        self.assertIn("Use tools only when needed", prompt)
        self.assertIn("If the user asks for one joke, one dog photo, or one trivia question, call that tool once and then answer.", prompt)
        self.assertIn("use city_to_coords before get_weather", prompt)
        self.assertIn("If the request is only for books, use only book_recs.", prompt)
        self.assertIn("include the fetched requested items", prompt)
        self.assertIn("Do not invent tools or unsupported capabilities.", prompt)

    def test_build_agent_options_uses_explicit_weekend_wizard_toolset(self) -> None:
        config = get_default_config()

        options = build_agent_options(config)

        self.assertEqual(options.allowed_tools, config.allowed_tool_names)
        self.assertEqual(options.max_turns, 4)
        self.assertEqual(str(options.cwd), str(config.cwd))
        self.assertIn("Weekend Wizard", options.system_prompt)

    async def test_run_claude_sdk_prompt_supports_dry_run_without_network(self) -> None:
        result = await run_claude_sdk_prompt("Tell me a joke.", dry_run=True)

        self.assertEqual(result["mode"], "dry-run")
        self.assertEqual(result["prompt"], "Tell me a joke.")
        self.assertEqual(result["config"]["mode"], "no-live-call")
        self.assertIn("allowed_tools", result["config"])
        self.assertIn("system_prompt", result["config"])
        self.assertIsNone(result["config"]["model"])
        self.assertEqual(result["config"]["allowed_tools"], get_default_config().allowed_tool_names)

    @patch("claude_sdk_agent.smoke.run_claude_sdk_prompt")
    async def test_run_smoke_scenarios_dry_run_reports_labels_prompts_and_config(self, mock_run_prompt) -> None:
        mock_run_prompt.side_effect = [
            {"mode": "dry-run", "prompt": "Tell me a joke.", "config": {"mode": "no-live-call"}},
            {
                "mode": "dry-run",
                "prompt": "Plan a cozy Saturday in New York with today's weather, 3 mystery books, one joke, and a dog pic.",
                "config": {"mode": "no-live-call"},
            },
        ]

        results = await run_smoke_scenarios(dry_run=True)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["label"], "joke-only")
        self.assertEqual(results[0]["prompt"], "Tell me a joke.")
        self.assertEqual(results[0]["mode"], "dry-run")
        self.assertIn("config", results[0])
        self.assertEqual(results[1]["label"], "weekend-multi-tool")
        self.assertIn("config", results[1])

    @patch("claude_sdk_agent.tools.l2_random_joke", return_value=JokeResult(joke="A fetched joke."))
    async def test_random_joke_tool_wraps_existing_l2_behavior(self, _mock_joke) -> None:
        result = await random_joke_tool.handler({})

        self.assertFalse(result["is_error"])
        self.assertEqual(json.loads(result["content"][0]["text"])["joke"], "A fetched joke.")

    @patch("claude_sdk_agent.tools.l2_random_dog", return_value={"status": "success", "image_url": "https://example.com/dog.jpg"})
    async def test_random_dog_tool_wraps_existing_l2_behavior(self, _mock_dog) -> None:
        result = await random_dog_tool.handler({})

        self.assertFalse(result["is_error"])
        self.assertEqual(json.loads(result["content"][0]["text"])["image_url"], "https://example.com/dog.jpg")

    @patch(
        "claude_sdk_agent.tools.l2_trivia",
        return_value={
            "category": "General Knowledge",
            "difficulty": "easy",
            "question": "What is 2+2?",
            "correct_answer": "4",
            "incorrect_answers": ["1", "2", "3"],
        },
    )
    async def test_trivia_tool_wraps_existing_l2_behavior(self, _mock_trivia) -> None:
        result = await trivia_tool.handler({})

        self.assertFalse(result["is_error"])
        self.assertEqual(json.loads(result["content"][0]["text"])["correct_answer"], "4")

    @patch(
        "claude_sdk_agent.tools.l2_book_recs",
        return_value={
            "topic": "mystery",
            "count": 1,
            "results": [{"title": "Dune", "author": "Frank Herbert"}],
        },
    )
    async def test_book_recs_tool_wraps_existing_l2_behavior(self, _mock_books) -> None:
        result = await book_recs_tool.handler({"topic": "mystery", "limit": 1})

        self.assertFalse(result["is_error"])
        payload = json.loads(result["content"][0]["text"])
        self.assertEqual(payload["topic"], "mystery")
        self.assertEqual(payload["count"], 1)

    @patch(
        "claude_sdk_agent.tools.l2_city_to_coords",
        return_value={"city": "New York", "latitude": 40.7128, "longitude": -74.0060},
    )
    async def test_city_to_coords_tool_wraps_existing_l2_behavior(self, _mock_geo) -> None:
        result = await city_to_coords_tool.handler({"city": "New York"})

        self.assertFalse(result["is_error"])
        self.assertEqual(json.loads(result["content"][0]["text"])["city"], "New York")

    @patch(
        "claude_sdk_agent.tools.l2_get_weather",
        return_value={"temperature": 22.5, "weather_summary": "mainly clear"},
    )
    async def test_get_weather_tool_wraps_existing_l2_behavior(self, _mock_weather) -> None:
        result = await get_weather_tool.handler({"latitude": 12.97, "longitude": 77.59})

        self.assertFalse(result["is_error"])
        self.assertEqual(json.loads(result["content"][0]["text"])["weather_summary"], "mainly clear")

    def test_build_sdk_server_exposes_only_weekend_wizard_tools(self) -> None:
        config = get_default_config()

        server = build_sdk_server(config)

        self.assertEqual(server["name"], config.server_name)
        self.assertEqual(server["type"], "sdk")
        tool_names = sorted(
            tool_def.name
            for tool_def in [
                random_joke_tool,
                random_dog_tool,
                trivia_tool,
                book_recs_tool,
                city_to_coords_tool,
                get_weather_tool,
            ]
        )
        self.assertEqual(
            tool_names,
            sorted(
                [
                    "random_joke",
                    "random_dog",
                    "trivia",
                    "book_recs",
                    "city_to_coords",
                    "get_weather",
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
