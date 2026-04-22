from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

import llm_client


class LlmClientTests(unittest.TestCase):
    def test_extract_json_handles_wrapped_text(self) -> None:
        parsed = llm_client.extract_json('Result: {"action":"finish","final_answer":"hi"} thanks')

        self.assertEqual(parsed["action"], "finish")
        self.assertEqual(parsed["final_answer"], "hi")

    @patch(
        "llm_client.call_model",
        return_value='{"thought":"Need a joke.","action":"tool","tool":"random_joke","args":{}}',
    )
    def test_llm_react_json_returns_model_json(self, _call_model: Mock) -> None:
        result = llm_client.llm_react_json([{"role": "user", "content": "hello"}], "demo-model")

        self.assertEqual(result["action"], "tool")
        self.assertEqual(result["tool"], "random_joke")

    @patch("llm_client.call_model", return_value='{"answer":"tightened"}')
    def test_llm_reflection_json_returns_model_json(self, _call_model: Mock) -> None:
        result = llm_client.llm_reflection_json([{"role": "user", "content": "hello"}], "demo-model")

        self.assertEqual(result["answer"], "tightened")

    @patch(
        "llm_client.call_model",
        side_effect=[
            '{"unexpected":"shape"}',
            '{"thought":"I can answer now.","action":"finish","final_answer":"Here is the answer."}',
        ],
    )
    def test_llm_react_json_repairs_schema_invalid_json(self, _call_model: Mock) -> None:
        result = llm_client.llm_react_json([{"role": "user", "content": "hello"}], "demo-model")

        self.assertEqual(result["action"], "finish")

    @patch("llm_client.call_model", side_effect=requests.RequestException("offline"))
    def test_llm_react_json_raises_when_model_request_fails_in_normal_mode(self, _call_model: Mock) -> None:
        with self.assertRaises(requests.RequestException):
            llm_client.llm_react_json([{"role": "user", "content": "hello"}], "demo-model")

    @patch(
        "llm_client.call_model",
        side_effect=[
            "not-json",
            requests.RequestException("repair failed"),
        ],
    )
    def test_llm_react_json_raises_when_repair_fails_in_normal_mode(self, _call_model: Mock) -> None:
        with self.assertRaises(requests.RequestException):
            llm_client.llm_react_json([{"role": "user", "content": "hello"}], "demo-model")

    @patch(
        "llm_client.call_model",
        side_effect=[
            '{"unexpected":"shape"}',
            '{"still":"wrong"}',
        ],
    )
    def test_llm_react_json_raises_when_schema_repair_still_fails(self, _call_model: Mock) -> None:
        with self.assertRaisesRegex(ValueError, "invalid ReAct decision JSON"):
            llm_client.llm_react_json([{"role": "user", "content": "hello"}], "demo-model")

    @patch(
        "llm_client.call_model",
        side_effect=[
            '{"unexpected":"shape"}',
            '{"answer":"tightened"}',
        ],
    )
    def test_llm_reflection_json_repairs_invalid_shape(self, _call_model: Mock) -> None:
        result = llm_client.llm_reflection_json([{"role": "user", "content": "hello"}], "demo-model")

        self.assertEqual(result["answer"], "tightened")

    @patch("llm_client.requests.get")
    @patch("llm_client.get_settings")
    def test_discover_model_uses_configured_model(self, mock_settings: Mock, mock_get: Mock) -> None:
        mock_settings.return_value = SimpleNamespace(
            ollama_url="http://127.0.0.1:11434/api/chat",
            preferred_models=("gpt-oss:20b-cloud",),
        )
        response = Mock()
        response.json.return_value = {
            "models": [
                {"name": "gpt-oss:20b-cloud"},
                {"name": "llama3.2:latest"},
            ]
        }
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        result = llm_client.discover_model(None)

        self.assertEqual(result, "gpt-oss:20b-cloud")

    @patch("llm_client.requests.get")
    @patch("llm_client.get_settings")
    def test_discover_model_fails_when_configured_model_is_missing(self, mock_settings: Mock, mock_get: Mock) -> None:
        mock_settings.return_value = SimpleNamespace(
            ollama_url="http://127.0.0.1:11434/api/chat",
            preferred_models=("gpt-oss:20b-cloud",),
        )
        response = Mock()
        response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "Configured Ollama model is not available"):
            llm_client.discover_model(None)


if __name__ == "__main__":
    unittest.main()
