from __future__ import annotations

"""Streamlit interface for Weekend Wizard."""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from config.config import get_settings
from logger.logging import get_logger
from schemas.api import ChatResponse, ReadinessResponse


logger = get_logger("agent.streamlit")

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
TRACE_LOG_PATH = Path("logs") / "trace.log"
TRACE_READ_BYTES = 1_000_000


@dataclass
class ChatTurn:
    """Rendered chat turn stored in Streamlit session state.

    Attributes:
        role: Chat role rendered in the UI, such as ``user`` or ``assistant``.
        content: Markdown content displayed for the chat turn.
        correlation_id: Optional app-wide correlation id for assistant responses.
        tool_observations: Optional serialized tool observations shown in the UI.
    """

    role: str
    content: str
    correlation_id: str | None = None
    tool_observations: list[dict[str, Any]] | None = None


def get_api_base_url() -> str:
    """Return the FastAPI base URL used by the Streamlit demo.

    Returns:
        The normalized backend base URL with any trailing slash removed.
    """
    return os.getenv("WEEKEND_WIZARD_API_URL", DEFAULT_API_BASE_URL).rstrip("/")


def build_chat_headers() -> dict[str, str]:
    """Return optional headers required by the backend chat endpoint."""
    api_key = get_settings().api_key
    if api_key is None:
        return {}
    return {"X-API-Key": api_key}


def load_readiness() -> ReadinessResponse:
    """Fetch readiness from the FastAPI backend.

    Returns:
        The validated readiness response returned by the backend.

    Raises:
        RuntimeError: If the backend is unreachable or returns invalid JSON.
    """
    base_url = get_api_base_url()
    try:
        response = requests.get(f"{base_url}/ready", timeout=10)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Weekend Wizard API at {base_url}. Start the API server first."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Weekend Wizard API returned an invalid readiness response.") from exc

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
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"prompt": prompt},
            headers=build_chat_headers(),
            timeout=get_settings().request_timeout,
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


def latest_assistant_turn() -> ChatTurn | None:
    """Return the latest assistant turn from the current Streamlit transcript."""
    for turn in reversed(st.session_state.get("chat_turns", [])):
        if turn.role == "assistant":
            return turn
    return None


def latest_user_prompt() -> str | None:
    """Return the latest user prompt from the current Streamlit transcript."""
    for turn in reversed(st.session_state.get("chat_turns", [])):
        if turn.role == "user":
            return turn.content
    return None


def load_trace_by_correlation_id(correlation_id: str) -> str | None:
    """Return the rendered backend trace block for one correlation id when available."""
    if not TRACE_LOG_PATH.exists():
        return None

    with TRACE_LOG_PATH.open("rb") as file:
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(max(0, size - TRACE_READ_BYTES))
        contents = file.read().decode("utf-8", errors="replace")

    marker = f"CORRELATION ID: {correlation_id}"
    marker_index = contents.rfind(marker)
    if marker_index == -1:
        return None

    block_start = contents.rfind("=" * 48, 0, marker_index)
    next_block_start = contents.find(f"\n{'=' * 48}\nCORRELATION ID:", marker_index)
    block_end = len(contents) if next_block_start == -1 else next_block_start
    return contents[max(0, block_start):block_end].strip()


def render_sidebar(readiness: ReadinessResponse) -> None:
    """Render Streamlit sidebar controls and backend details."""
    with st.sidebar:
        st.title("Weekend Wizard")
        st.caption("Operable weekend-planning agent.")
        st.divider()
        st.header("Runtime")
        st.write(f"API: `{get_api_base_url()}`")
        st.write(f"Status: `{readiness.status}`")
        st.write(f"Provider: `{readiness.provider}`")
        st.write(f"Model: `{readiness.model_name}`")
        st.write(f"Tools: `{readiness.tool_count}`")
        with st.expander("Readiness diagnostics"):
            st.write(f"Provider reachable: `{readiness.checks.provider_reachable}`")
            st.write(f"MCP session ready: `{readiness.checks.mcp_session_ready}`")
            st.write(f"Auth configured: `{readiness.checks.auth_configured}`")
            st.write(f"Rate limit: `{readiness.rate_limit_requests}/{readiness.rate_limit_window_seconds}s`")
            st.write(f"Request timeout: `{readiness.request_timeout_seconds}s`")
            st.write(f"Trace logging: `{readiness.checks.trace_logging_configured}`")
        if st.button("New Chat", use_container_width=True):
            reset_chat()
            st.rerun()


def render_chat_history() -> None:
    """Render accumulated Streamlit chat history."""
    for turn in st.session_state.get("chat_turns", []):
        with st.chat_message(turn.role):
            st.markdown(turn.content)
            if turn.correlation_id:
                with st.expander("Agent run details"):
                    st.write(f"Correlation ID: `{turn.correlation_id}`")
                    st.write(f"Tool observations: `{len(turn.tool_observations or [])}`")
            if turn.tool_observations:
                with st.expander("Tool observations"):
                    for observation in turn.tool_observations:
                        st.code(
                            f"{observation['tool_name']} args={observation['args']}\n{observation['payload']}",
                            language="json",
                        )


def render_observability(readiness: ReadinessResponse) -> None:
    """Render production diagnostics for the latest Streamlit interaction."""
    latest_turn = latest_assistant_turn()
    st.subheader("Runtime Posture")
    st.write(f"Provider: `{readiness.provider}`")
    st.write(f"Model: `{readiness.model_name}`")
    st.write(f"Tools discovered: `{readiness.tool_count}`")
    st.write(f"Auth configured: `{readiness.checks.auth_configured}`")
    st.write(f"Rate limit: `{readiness.rate_limit_requests}/{readiness.rate_limit_window_seconds}s`")
    st.write(f"Trace logging: `{readiness.checks.trace_logging_configured}`")

    st.subheader("Last Run")
    if latest_turn is None or latest_turn.correlation_id is None:
        st.info("Run a chat request to see request-level observability.")
        return

    st.write(f"Correlation ID: `{latest_turn.correlation_id}`")
    st.write(f"Tool observations: `{len(latest_turn.tool_observations or [])}`")
    st.write(f"Answer length: `{len(latest_turn.content)}`")

    prompt = latest_user_prompt()
    if prompt:
        st.subheader("Replay Request")
        header_line = '  -Headers @{"X-API-Key"="<configured>"} `\n' if get_settings().api_key else ""
        body = json.dumps({"prompt": prompt})
        st.code(
            "Invoke-RestMethod -Method Post `\n"
            f"  -Uri {get_api_base_url()}/chat `\n"
            f"{header_line}"
            '  -ContentType "application/json" `\n'
            f"  -Body '{body}'",
            language="powershell",
        )

    st.subheader("Backend Trace")
    trace_block = load_trace_by_correlation_id(latest_turn.correlation_id)
    if trace_block is None:
        st.warning("Trace block was not found in logs/trace.log yet.")
        return
    st.code(trace_block, language="text")


def append_result(result: ChatResponse) -> None:
    """Append one assistant result to the Streamlit transcript."""
    st.session_state.chat_turns.append(
        ChatTurn(
            role="assistant",
            content=result.answer,
            correlation_id=result.correlation_id,
            tool_observations=[observation.model_dump() for observation in result.tool_observations],
        )
    )


def render_chat_tab(readiness: ReadinessResponse) -> None:
    """Render the main chat workflow."""
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
            with st.expander("Agent run details"):
                st.write(f"Correlation ID: `{result.correlation_id}`")
                st.write(f"Tool observations: `{len(result.tool_observations)}`")
            if result.tool_observations:
                with st.expander("Tool observations"):
                    for observation in result.tool_observations:
                        st.code(
                            f"{observation.tool_name} args={observation.args}\n{observation.payload}",
                            language="json",
                        )
    append_result(result)


def run_app() -> None:
    """Render the Streamlit Weekend Wizard interface."""
    st.set_page_config(page_title="Weekend Wizard", page_icon="W", layout="wide")

    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []

    try:
        readiness = load_readiness()
    except Exception as exc:
        logger.exception("Streamlit readiness check failed: %s", exc)
        st.error(str(exc))
        return

    render_sidebar(readiness)
    chat_tab, observability_tab = st.tabs(["Chat", "Observability"])
    with chat_tab:
        render_chat_tab(readiness)
    with observability_tab:
        render_observability(readiness)


if __name__ == "__main__":
    run_app()
