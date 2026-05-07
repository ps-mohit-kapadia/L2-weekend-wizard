from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from claude_sdk_agent.config import get_default_config
from claude_sdk_agent.runner import build_agent_options, run_claude_sdk_prompt
from claude_sdk_agent.tools import random_joke_tool


class ClaudeSdkAgentTests(unittest.IsolatedAsyncioTestCase):
    def test_build_agent_options_uses_explicit_allowed_joke_tool(self) -> None:
        config = get_default_config()

        options = build_agent_options(config)

        self.assertEqual(options.allowed_tools, [config.joke_tool_name])
        self.assertEqual(options.max_turns, 4)
        self.assertEqual(str(options.cwd), str(config.cwd))

    async def test_run_claude_sdk_prompt_supports_dry_run_without_network(self) -> None:
        result = await run_claude_sdk_prompt("Tell me a joke.", dry_run=True)

        self.assertEqual(result["mode"], "dry-run")
        self.assertEqual(result["prompt"], "Tell me a joke.")
        self.assertIn("allowed_tools", result["config"])

    @patch("claude_sdk_agent.tools.l2_random_joke", return_value={"joke": "A fetched joke."})
    async def test_random_joke_tool_wraps_existing_l2_behavior(self, _mock_joke) -> None:
        result = await random_joke_tool.handler({})

        self.assertFalse(result["is_error"])
        self.assertEqual(json.loads(result["content"][0]["text"])["joke"], "A fetched joke.")


if __name__ == "__main__":
    unittest.main()
