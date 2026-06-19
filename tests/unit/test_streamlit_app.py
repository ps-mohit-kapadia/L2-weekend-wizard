from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

import streamlit_app


class StreamlitAppTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_get_api_base_url_uses_default(self) -> None:
        self.assertEqual(streamlit_app.get_api_base_url(), "http://127.0.0.1:8000")

    @patch.dict("os.environ", {"WEEKEND_WIZARD_API_URL": "http://example.com/"}, clear=True)
    def test_get_api_base_url_strips_trailing_slash(self) -> None:
        self.assertEqual(streamlit_app.get_api_base_url(), "http://example.com")

    @patch("streamlit_app.requests.get")
    def test_load_readiness_returns_structured_response(self, mock_get: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "status": "ready",
            "model_name": "llama3.2:latest",
            "tool_count": 2,
            "checks": {
                "model_resolved": True,
                "model_available": True,
                "server_path_exists": True,
                "ollama_reachable": True,
                "mcp_session_ready": True,
                "tools_discovered": True,
            },
            "details": None,
        }
        mock_get.return_value = response

        readiness = streamlit_app.load_readiness()

        self.assertEqual(readiness.status, "ready")
        self.assertEqual(readiness.tool_count, 2)

    @patch("streamlit_app.requests.get", side_effect=requests.RequestException("offline"))
    def test_load_readiness_raises_when_api_is_unreachable(self, _mock_get: Mock) -> None:
        with self.assertRaises(RuntimeError):
            streamlit_app.load_readiness()

    @patch("streamlit_app.requests.post")
    @patch("streamlit_app.get_settings")
    def test_send_chat_prompt_returns_structured_response(self, mock_settings: Mock, mock_post: Mock) -> None:
        mock_settings.return_value = SimpleNamespace(request_timeout=888, api_key=None)
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "correlation_id": "corr_1234567890abcdef",
            "answer": "Weekend plan ready.",
            "tool_observations": [],
        }
        mock_post.return_value = response

        result = streamlit_app.send_chat_prompt("hello")

        self.assertEqual(result.correlation_id, "corr_1234567890abcdef")
        self.assertEqual(result.answer, "Weekend plan ready.")
        self.assertEqual(result.tool_observations, [])
        self.assertEqual(mock_post.call_args.kwargs["timeout"], 888)
        self.assertEqual(mock_post.call_args.kwargs["headers"], {})

    @patch("streamlit_app.requests.post")
    @patch("streamlit_app.get_settings")
    def test_send_chat_prompt_sends_configured_api_key(self, mock_settings: Mock, mock_post: Mock) -> None:
        mock_settings.return_value = SimpleNamespace(request_timeout=888, api_key="secret")
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "correlation_id": "corr_1234567890abcdef",
            "answer": "Weekend plan ready.",
            "tool_observations": [],
        }
        mock_post.return_value = response

        streamlit_app.send_chat_prompt("hello")

        self.assertEqual(mock_post.call_args.kwargs["headers"], {"X-API-Key": "secret"})

    @patch("streamlit_app.requests.post")
    @patch("streamlit_app.get_settings")
    def test_send_chat_prompt_raises_with_api_error_detail(self, mock_settings: Mock, mock_post: Mock) -> None:
        mock_settings.return_value = SimpleNamespace(request_timeout=888, api_key=None)
        response = Mock()
        response.status_code = 503
        response.json.return_value = {"detail": "Service is not ready."}
        mock_post.return_value = response

        with self.assertRaisesRegex(RuntimeError, "Service is not ready."):
            streamlit_app.send_chat_prompt("hello")


if __name__ == "__main__":
    unittest.main()
