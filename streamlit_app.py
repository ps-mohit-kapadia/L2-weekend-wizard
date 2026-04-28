from __future__ import annotations

"""Streamlit interface for Weekend Wizard."""

import os
from dataclasses import dataclass
from typing import Any

import requests
import streamlit as st

from config.config import get_settings
from logger.logging import get_logger
from schemas.api import ChatResponse, ReadinessResponse


logger = get_logger("agent.streamlit")

API_KEY_HEADER = "X-API-Key"


@dataclass
class ChatTurn:
    """Rendered chat turn stored in Streamlit session state.

    Attributes:
        role: Chat role rendered in the UI, such as ``user`` or ``assistant``.
        content: Markdown content displayed for the chat turn.
        tool_observations: Optional serialized tool observations shown in the UI.
    """

    role: str
    content: str
    tool_observations: list[dict[str, Any]] | None = None


def get_api_base_url() -> str:
    """Return the FastAPI base URL used by the Streamlit demo.

    Returns:
        The normalized backend base URL with any trailing slash removed.
    """
    configured_url = os.getenv("WEEKEND_WIZARD_API_URL")
    if configured_url:
        return configured_url.rstrip("/")
    return get_settings().api_url


def get_api_headers() -> dict[str, str]:
    """Return required headers for backend API requests."""
    api_key = os.getenv("WEEKEND_WIZARD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("WEEKEND_WIZARD_API_KEY is not configured for the Streamlit client.")
    return {API_KEY_HEADER: api_key}


def load_readiness() -> ReadinessResponse:
    """Fetch readiness from the FastAPI backend.

    Returns:
        The validated readiness response returned by the backend.

    Raises:
        RuntimeError: If the backend is unreachable, returns invalid JSON, or
            responds with an error status.
    """
    base_url = get_api_base_url()
    headers = get_api_headers()
    try:
        response = requests.get(f"{base_url}/ready", headers=headers, timeout=10)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Weekend Wizard API at {base_url}. Start the API server first."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Weekend Wizard API returned an invalid readiness response.") from exc

    if response.status_code != 200:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        raise RuntimeError(detail or f"Weekend Wizard API returned HTTP {response.status_code}.")

    return ReadinessResponse.model_validate(payload)


def send_chat_prompt(prompt: str) -> ChatResponse:
    """Send one chat prompt to the FastAPI backend.

    Args:
        prompt: User prompt to send to the backend.

    Returns:
        The validated structured chat response.

    Raises:
        RuntimeError: If the backend is unreachable, returns invalid JSON, or
            responds with an error status.
    """
    base_url = get_api_base_url()
    headers = get_api_headers()
    request_timeout = get_settings().request_timeout
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"prompt": prompt},
            headers=headers,
            timeout=request_timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Weekend Wizard API at {base_url}. Start the API server first."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Weekend Wizard API returned an invalid chat response.") from exc

    if response.status_code != 200:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        raise RuntimeError(detail or f"Weekend Wizard API returned HTTP {response.status_code}.")

    return ChatResponse.model_validate(payload)


def reset_chat() -> None:
    """Reset the current Streamlit chat transcript."""
    st.session_state.chat_turns = []
    logger.info("Reset Streamlit chat history")


def render_sidebar(readiness: ReadinessResponse) -> None:
    """Render Streamlit sidebar controls and backend details."""
    with st.sidebar:
        st.header("Runtime")
        st.write(f"API: `{get_api_base_url()}`")
        st.write(f"Status: `{readiness.status}`")
        st.write(f"Model: `{readiness.model_name}`")
        st.write(f"Tools: `{readiness.tool_count}`")
        if st.button("New Chat", use_container_width=True):
            reset_chat()
            st.rerun()


def render_chat_history() -> None:
    """Render accumulated Streamlit chat history."""
    for turn in st.session_state.get("chat_turns", []):
        with st.chat_message(turn.role):
            st.markdown(turn.content)
            if turn.tool_observations:
                with st.expander("Tool observations"):
                    for observation in turn.tool_observations:
                        st.code(
                            f"{observation['tool_name']} args={observation['args']}\n{observation['payload']}",
                            language="json",
                        )

def append_result(result: ChatResponse) -> None:
    """Append one assistant result to the Streamlit transcript."""
    st.session_state.chat_turns.append(
        ChatTurn(
            role="assistant",
            content=result.answer,
            tool_observations=[observation.model_dump() for observation in result.tool_observations],
        )
    )


def run_app() -> None:
    """Render the Streamlit Weekend Wizard interface."""
    st.set_page_config(page_title="Weekend Wizard", page_icon="W", layout="wide")
    st.title("Weekend Wizard")
    st.caption("Plan your weekend with a Streamlit demo backed by the FastAPI service.")

    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []

    try:
        readiness = load_readiness()
    except Exception as exc:
        logger.exception("Streamlit readiness check failed: %s", exc)
        st.error(str(exc))
        return

    render_sidebar(readiness)
    if readiness.status != "ready":
        st.error(readiness.details or "Weekend Wizard API is not ready.")
        return

    render_chat_history()

    prompt = st.chat_input("What kind of weekend are you looking for?")
    if not prompt:
        return

    st.session_state.chat_turns.append(ChatTurn(role="user", content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Planning your weekend..."):
            try:
                logger.info("Sending Streamlit prompt with length %d", len(prompt))
                result = send_chat_prompt(prompt)
                logger.info(
                    "Streamlit prompt completed with %d observations and answer length=%d",
                    len(result.tool_observations),
                    len(result.answer),
                )
            except Exception as exc:
                logger.exception("Streamlit interaction failed: %s", exc)
                st.error(str(exc))
                return

            st.markdown(result.answer)
            if result.tool_observations:
                with st.expander("Tool observations"):
                    for observation in result.tool_observations:
                        st.code(
                            f"{observation.tool_name} args={observation.args}\n{observation.payload}",
                            language="json",
                        )
    append_result(result)


if __name__ == "__main__":
    run_app()
