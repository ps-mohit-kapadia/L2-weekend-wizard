from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

import llm_client
from logger.tracing.request_trace import create_trace


class LlmClientTests(unittest.TestCase):
    @patch("llm_client.requests.post")
    @patch("llm_client.get_settings")
    def test_call_model_uses_request_timeout_setting(self, mock_settings: Mock, mock_post: Mock) -> None:
        mock_settings.return_value = SimpleNamespace(
            llm_provider="ollama",
            ollama_url="http://127.0.0.1:11434/api/chat",
            request_timeout=777,
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"message": {"content": '{"answer":"ok"}'}}
        mock_post.return_value = response

        trace = create_trace("hello")
        llm_client.call_model(
            [{"role": "user", "content": "hello"}],
            "demo-model",
            temperature=0.2,
            trace=trace,
        )

        mock_post.assert_called_once()
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 777)
        self.assertEqual(trace.events[-2].event, "llm_call_started")
        self.assertEqual(trace.events[-1].event, "llm_call_completed")

    @patch("llm_client.requests.post")
    @patch("llm_client.get_settings")
    def test_call_model_uses_aiplatform_endpoint_and_timeout(self, mock_settings: Mock, mock_post: Mock) -> None:
        mock_settings.return_value = SimpleNamespace(
            llm_provider="aiplatform",
            aiplatform_api_key="secret:public",
            aiplatform_base_url="https://aiapidev.3ecompany.com",
            aiplatform_chat_path="/v1/chat/completions",
            aiplatform_timeout=123,
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": '{"answer":"ok"}'}}]
        }
        mock_post.return_value = response

        llm_client.call_model(
            [{"role": "user", "content": "hello"}],
            "baseten/deepseek-ai/deepseek-v3.1",
            temperature=0.0,
            json_mode=True,
        )

        mock_post.assert_called_once()
        self.assertEqual(
            mock_post.call_args.args[0],
            "https://aiapidev.3ecompany.com/v1/chat/completions",
        )
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 123)
        self.assertEqual(
            mock_post.call_args.kwargs["headers"]["Authorization"],
            "Bearer secret:public",
        )
        self.assertEqual(
            mock_post.call_args.kwargs["json"]["response_format"],
            {"type": "json_object"},
        )

    def test_extract_json_handles_wrapped_text(self) -> None:
        parsed = llm_client.extract_json('Result: {"action":"finish","final_answer":"hi"} thanks')

        self.assertEqual(parsed["action"], "finish")
        self.assertEqual(parsed["final_answer"], "hi")

    @patch(
        "llm_client.call_model",
        return_value='{"thought":"Need a joke.","action":"tool","tool":"random_joke","args":{}}',
    )
    def test_llm_react_json_returns_model_json(self, _call_model: Mock) -> None:
        result = llm_client.llm_react_json(
            [{"role": "user", "content": "hello"}],
            "demo-model",
            allowed_tools=["random_joke"],
        )

        self.assertEqual(result.action, "tool")
        self.assertEqual(result.tool, "random_joke")

    @patch("llm_client.call_model", return_value='{"verdict":"pass","intro":"Nice.","outro":"Enjoy.","issues":[]}')
    def test_llm_reflection_json_returns_model_json(self, _call_model: Mock) -> None:
        result = llm_client.llm_reflection_json([{"role": "user", "content": "hello"}], "demo-model")

        self.assertEqual(result.verdict, "pass")
        self.assertEqual(result.intro, "Nice.")

    @patch(
        "llm_client.call_model",
        side_effect=[
            '{"unexpected":"shape"}',
            '{"thought":"I can answer now.","action":"finish","final_answer":"Here is the answer."}',
        ],
    )
    def test_llm_react_json_repairs_schema_invalid_json(self, _call_model: Mock) -> None:
        result = llm_client.llm_react_json(
            [{"role": "user", "content": "hello"}],
            "demo-model",
            allowed_tools=["random_joke"],
        )

        self.assertEqual(result.action, "finish")

    @patch("llm_client.call_model", side_effect=requests.RequestException("offline"))
    def test_llm_react_json_raises_when_model_request_fails_in_normal_mode(self, _call_model: Mock) -> None:
        with self.assertRaises(requests.RequestException):
            llm_client.llm_react_json(
                [{"role": "user", "content": "hello"}],
                "demo-model",
                allowed_tools=["random_joke"],
            )

    @patch(
        "llm_client.call_model",
        side_effect=[
            "not-json",
            requests.RequestException("repair failed"),
        ],
    )
    def test_llm_react_json_raises_when_repair_fails_in_normal_mode(self, _call_model: Mock) -> None:
        with self.assertRaises(requests.RequestException):
            llm_client.llm_react_json(
                [{"role": "user", "content": "hello"}],
                "demo-model",
                allowed_tools=["random_joke"],
            )

    @patch(
        "llm_client.call_model",
        side_effect=[
            '{"unexpected":"shape"}',
            '{"still":"wrong"}',
        ],
    )
    def test_llm_react_json_raises_when_schema_repair_still_fails(self, _call_model: Mock) -> None:
        with self.assertRaisesRegex(ValueError, "invalid ReAct decision JSON"):
            llm_client.llm_react_json(
                [{"role": "user", "content": "hello"}],
                "demo-model",
                allowed_tools=["random_joke"],
            )

    @patch(
        "llm_client.call_model",
        side_effect=[
            '{"unexpected":"shape"}',
            '{"thought":"I have enough information.","action":"finish","final_answer":"Here is your joke."}',
        ],
    )
    def test_llm_react_json_repair_includes_original_context_and_allowed_tools(self, mock_call_model: Mock) -> None:
        messages = [
            {"role": "system", "content": "react rules"},
            {
                "role": "user",
                "content": "Structured observation summary:\n- random_joke: fetched one joke",
            },
            {"role": "user", "content": "Tell me a joke."},
        ]

        llm_client.llm_react_json(messages, "demo-model", allowed_tools=["random_joke", "random_dog"])

        repair_messages = mock_call_model.call_args_list[1].args[0]
        repair_system = repair_messages[0]["content"]
        repair_user = repair_messages[-1]["content"]

        self.assertIn("Allowed tools:", repair_system)
        self.assertIn("- random_joke", repair_system)
        self.assertIn("- random_dog", repair_system)
        self.assertIn("Never invent tools.", repair_system)
        self.assertIn("If prior observations already satisfy the request, return finish.", repair_system)
        self.assertIn("one successful random_joke, random_dog, or trivia observation already satisfies the request", repair_system)
        self.assertIn("Structured observation summary:", str(repair_messages))
        self.assertIn("- random_joke: fetched one joke", str(repair_messages))
        self.assertNotIn('[tool:random_joke] {"joke":"A fetched joke."}', str(repair_messages))
        self.assertIn("Invalid output:", repair_user)

    @patch(
        "llm_client.call_model",
        side_effect=[
            '{"unexpected":"shape"}',
            '{"verdict":"pass","intro":"Nice.","outro":"","issues":[]}',
        ],
    )
    def test_llm_reflection_json_repairs_invalid_shape(self, _call_model: Mock) -> None:
        result = llm_client.llm_reflection_json([{"role": "user", "content": "hello"}], "demo-model")

        self.assertEqual(result.verdict, "pass")
        self.assertEqual(result.intro, "Nice.")

    @patch("llm_client.requests.get")
    @patch("llm_client.get_settings")
    def test_discover_model_uses_configured_model(self, mock_settings: Mock, mock_get: Mock) -> None:
        mock_settings.return_value = SimpleNamespace(
            ollama_url="http://127.0.0.1:11434/api/chat",
            preferred_models=("gpt-oss:20b-cloud",),
            llm_provider="ollama",
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
            llm_provider="ollama",
        )
        response = Mock()
        response.json.return_value = {"models": [{"name": "llama3.2:latest"}]}
        response.raise_for_status.return_value = None
        mock_get.return_value = response

        with self.assertRaisesRegex(RuntimeError, "Configured Ollama model is not available"):
            llm_client.discover_model(None)

    @patch("llm_client.get_settings")
    def test_discover_model_returns_configured_aiplatform_model_without_ollama_lookup(
        self, mock_settings: Mock
    ) -> None:
        mock_settings.return_value = SimpleNamespace(
            llm_provider="aiplatform",
            preferred_models=("baseten/deepseek-ai/deepseek-v3.1",),
        )

        result = llm_client.discover_model(None)

        self.assertEqual(result, "baseten/deepseek-ai/deepseek-v3.1")


if __name__ == "__main__":
    unittest.main()
