from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

import streamlit_app


class StreamlitAppTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_get_api_base_url_uses_default(self) -> None:
        with patch("streamlit_app.get_settings") as mock_settings:
            mock_settings.return_value.api_url = "http://127.0.0.1:8000"
            self.assertEqual(streamlit_app.get_api_base_url(), "http://127.0.0.1:8000")

    @patch.dict("os.environ", {"WEEKEND_WIZARD_API_URL": "http://example.com/"}, clear=True)
    def test_get_api_base_url_strips_trailing_slash(self) -> None:
        self.assertEqual(streamlit_app.get_api_base_url(), "http://example.com")

    @patch.dict("os.environ", {"WEEKEND_WIZARD_API_KEY": "secret-key"}, clear=True)
    def test_get_api_headers_returns_configured_api_key(self) -> None:
        self.assertEqual(streamlit_app.get_api_headers(), {"X-API-Key": "secret-key"})

    @patch.dict("os.environ", {}, clear=True)
    def test_get_api_headers_raises_when_api_key_is_missing(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "WEEKEND_WIZARD_API_KEY"):
            streamlit_app.get_api_headers()

    @patch.dict("os.environ", {"WEEKEND_WIZARD_API_KEY": "secret-key"}, clear=True)
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

        with patch("streamlit_app.get_settings") as mock_settings:
            mock_settings.return_value.api_url = "http://127.0.0.1:8000"
            readiness = streamlit_app.load_readiness()

        self.assertEqual(readiness.status, "ready")
        self.assertEqual(readiness.tool_count, 2)
        mock_get.assert_called_once_with(
            "http://127.0.0.1:8000/ready",
            headers={"X-API-Key": "secret-key"},
            timeout=10,
        )

    @patch.dict("os.environ", {"WEEKEND_WIZARD_API_KEY": "secret-key"}, clear=True)
    @patch("streamlit_app.requests.get", side_effect=requests.RequestException("offline"))
    def test_load_readiness_raises_when_api_is_unreachable(self, _mock_get: Mock) -> None:
        with patch("streamlit_app.get_settings") as mock_settings:
            mock_settings.return_value.api_url = "http://127.0.0.1:8000"
            with self.assertRaises(RuntimeError):
                streamlit_app.load_readiness()

    @patch.dict("os.environ", {"WEEKEND_WIZARD_API_KEY": "secret-key"}, clear=True)
    @patch("streamlit_app.requests.post")
    def test_send_chat_prompt_returns_structured_response(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "answer": "Weekend plan ready.",
            "tool_observations": [],
            "used_fallback": False,
        }
        mock_post.return_value = response

        with patch("streamlit_app.get_settings") as mock_settings:
            mock_settings.return_value.api_url = "http://127.0.0.1:8000"
            result = streamlit_app.send_chat_prompt("hello")

        self.assertEqual(result.answer, "Weekend plan ready.")
        self.assertEqual(result.tool_observations, [])
        self.assertFalse(result.used_fallback)
        mock_post.assert_called_once_with(
            "http://127.0.0.1:8000/chat",
            json={"prompt": "hello"},
            headers={"X-API-Key": "secret-key"},
            timeout=streamlit_app.CHAT_REQUEST_TIMEOUT_SECONDS,
        )

    @patch.dict("os.environ", {"WEEKEND_WIZARD_API_KEY": "secret-key"}, clear=True)
    @patch("streamlit_app.requests.post")
    def test_send_chat_prompt_raises_with_api_error_detail(self, mock_post: Mock) -> None:
        response = Mock()
        response.status_code = 503
        response.json.return_value = {"detail": "Service is not ready."}
        mock_post.return_value = response

        with patch("streamlit_app.get_settings") as mock_settings:
            mock_settings.return_value.api_url = "http://127.0.0.1:8000"
            with self.assertRaisesRegex(RuntimeError, "Service is not ready."):
                streamlit_app.send_chat_prompt("hello")


if __name__ == "__main__":
    unittest.main()
