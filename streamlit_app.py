from __future__ import annotations

"""Streamlit interface for Weekend Wizard."""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import streamlit as st

from agent.prompts import PROMPT_CONTRACTS, build_react_messages, build_reflection_messages
from config.config import get_settings
from evals.runner import EvalCase, EvalResult, evaluate_case, load_cases, post_chat
from logger.logging import get_logger
from schemas.api import ChatResponse, ReadinessResponse


logger = get_logger("agent.streamlit")

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
    return get_settings().api_url.rstrip("/")


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


def extract_reflection_summary(trace_block: str) -> dict[str, str] | None:
    """Extract the latest reflection review summary from a rendered trace block."""
    lines = trace_block.splitlines()
    for index, line in reversed(list(enumerate(lines))):
        if "EVENT: reflection.reviewed" not in line:
            continue
        summary: dict[str, str] = {}
        for detail in lines[index + 1:]:
            if detail.startswith("[") or not detail.startswith("* "):
                break
            key, _, value = detail[2:].partition(": ")
            summary[key] = value
        return summary
    return None


def extract_prompt_provenance(trace_block: str) -> list[dict[str, str]]:
    """Extract prompt usage receipts from a rendered trace block."""
    usage: list[dict[str, str]] = []
    lines = trace_block.splitlines()
    for index, line in enumerate(lines):
        if "EVENT: prompt.used" not in line:
            continue
        receipt: dict[str, str] = {}
        for detail in lines[index + 1:]:
            if detail.startswith("[") or not detail.startswith("* "):
                break
            key, _, value = detail[2:].partition(": ")
            receipt[key] = value
        usage.append(receipt)
    return usage


def summarize_prompt_usage(receipts: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Group prompt usage receipts by prompt id for display."""
    summary: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        prompt_id = receipt.get("prompt_id", "unknown")
        item = summary.setdefault(
            prompt_id,
            {
                "count": 0,
                "version": receipt.get("prompt_version", "?"),
                "contract": receipt.get("output_contract", "unknown"),
                "spans": [],
            },
        )
        item["count"] += 1
        if receipt.get("span_id"):
            item["spans"].append(receipt["span_id"])
    return summary


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

    reflection = extract_reflection_summary(trace_block)
    if reflection:
        st.subheader("Reflection")
        st.write(f"Verdict: `{reflection.get('verdict', 'unknown')}`")
        st.write(f"Issues: `{reflection.get('issues_count', '0')}`")
        if reflection.get("intro_preview"):
            st.write(f"Intro: {reflection['intro_preview']}")
        if reflection.get("outro_preview"):
            st.write(f"Outro: {reflection['outro_preview']}")

    prompt_usage = extract_prompt_provenance(trace_block)
    if prompt_usage:
        st.subheader("Prompt Provenance")
        for receipt in prompt_usage:
            st.write(
                f"`{receipt.get('prompt_id', 'unknown')}` "
                f"v{receipt.get('prompt_version', '?')} -> "
                f"`{receipt.get('output_contract', 'unknown')}`"
            )


def render_prompts_tab() -> None:
    """Render read-only prompt registry metadata."""
    st.subheader("Prompt Governance")
    st.caption(
        "Prompts are governed production contracts. Runtime editing is intentionally disabled for release safety."
    )

    st.markdown(
        "- Prompt identity, version, phase, and output contract are recorded on every LLM trace span.\n"
        "- Prompt text remains code-owned for this release; UI editing is a future hardening phase.\n"
        "- Invalid prompt outputs are contract-validated, repaired once, then fail closed or fall back safely."
    )

    latest_turn = latest_assistant_turn()
    prompt_usage: dict[str, dict[str, Any]] = {}
    if latest_turn and latest_turn.correlation_id:
        trace_block = load_trace_by_correlation_id(latest_turn.correlation_id)
        if trace_block:
            prompt_usage = summarize_prompt_usage(extract_prompt_provenance(trace_block))

    st.subheader("Prompt Contracts")
    for contract in PROMPT_CONTRACTS:
        usage = prompt_usage.get(contract.prompt_id)
        with st.expander(f"{contract.prompt_id} v{contract.version}", expanded=contract.prompt_id in {"react_planner", "reflection_review"}):
            st.write(f"Purpose: {contract.purpose}")
            st.write(f"Owns: {contract.owns}")
            st.write(f"Does not own: {contract.does_not_own}")
            st.write(f"Output contract: `{contract.output_contract}`")
            st.write(f"Failure mode: {contract.failure_mode}")
            st.write("Runtime editing: `disabled`")
            if usage:
                spans = ", ".join(usage["spans"]) if usage["spans"] else "n/a"
                st.write(f"Latest trace usage: `{usage['count']}` call(s), spans `{spans}`")
            else:
                st.write("Latest trace usage: `not used in latest run`")

    st.subheader("Prompt Template Preview")
    react_preview = build_react_messages(
        planner_messages=[],
        tool_names=["city_to_coords", "get_weather", "book_recs", "random_joke", "random_dog", "trivia"],
        step_number=1,
        max_steps=6,
    )[0]["content"]
    reflection_preview = build_reflection_messages(
        "Sample request only. No live user data is shown here.",
        ["- Weather: sample location: 21°C, clear sky"],
        "Weather for sample location: 21°C, clear sky.",
    )[0]["content"]
    with st.expander("ReAct Planner Prompt Template"):
        st.code(react_preview, language="text")
    with st.expander("Reflection Review Prompt Template"):
        st.code(reflection_preview, language="text")


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

    prompt = st.session_state.pop("pending_prompt", None)
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


def render_eval_tab() -> None:
    """Render the evaluation tab for running evals from the UI."""
    st.subheader("Evaluation Framework")
    st.caption(
        "Run acceptance evals against the Weekend Wizard API to verify tool usage and response quality."
    )

    st.markdown(
        "- Evals verify that the agent uses the correct tools for each prompt type.\n"
        "- Each case checks expected tools, forbidden tools, and answer markers.\n"
        "- Results show pass/fail status with detailed failure reasons.\n"
        "- Requires the API to be running at the configured base URL."
    )

    cases_path = Path(__file__).resolve().parent / "evals" / "cases.jsonl"
    api_url = get_api_base_url()
    timeout = get_settings().request_timeout

    try:
        cases = load_cases(cases_path)
    except Exception as exc:
        st.error(f"Failed to load eval cases: {exc}")
        return

    st.subheader("Eval Cases")
    st.write(f"Loaded {len(cases)} eval cases from `evals/cases.jsonl`")

    case_data = []
    for case in cases:
        prompt_preview = case.prompt[:60] + "..." if len(case.prompt) > 60 else case.prompt
        expected = ", ".join(case.expected_tools) if case.expected_tools else "-"
        case_data.append({
            "Case ID": case.id,
            "Prompt Preview": prompt_preview,
            "Expected Tools": expected,
            "Max Observations": case.max_observations,
        })

    st.dataframe(case_data, use_container_width=True, hide_index=True)

    st.subheader("Run Evals")
    col1, col2 = st.columns([1, 3])
    with col1:
        run_button = st.button("Run All Evals", type="primary", use_container_width=True)

    if "eval_results" not in st.session_state:
        st.session_state.eval_results = None

    if run_button:
        results: list[EvalResult] = []
        progress_bar = st.progress(0, text="Running evals...")
        status_text = st.empty()

        for index, case in enumerate(cases):
            status_text.text(f"Running case {index + 1}/{len(cases)}: {case.id}...")
            try:
                payload = post_chat(api_url, case.prompt, timeout)
                result = evaluate_case(case, payload)
            except Exception as exc:
                result = EvalResult(
                    case_id=case.id,
                    passed=False,
                    failures=[str(exc)],
                    observed_tools=[],
                    answer_preview="",
                )
            results.append(result)
            progress_bar.progress((index + 1) / len(cases), text=f"Running evals... ({index + 1}/{len(cases)})")

        progress_bar.empty()
        status_text.empty()
        st.session_state.eval_results = results
        st.rerun()

    results = st.session_state.eval_results
    if results is None:
        st.info("Click 'Run All Evals' to execute the evaluation suite.")
        return

    st.subheader("Results Summary")
    passed_count = sum(1 for result in results if result.passed)
    failed_count = len(results) - passed_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cases", len(results))
    with col2:
        st.metric("Passed", passed_count, delta_color="normal")
    with col3:
        st.metric("Failed", failed_count, delta_color="inverse" if failed_count > 0 else "normal")

    st.subheader("Detailed Results")
    for result in results:
        with st.expander(
            f"{'PASS' if result.passed else 'FAIL'}: {result.case_id}",
            expanded=not result.passed,
        ):
            if result.passed:
                st.success("All checks passed.")
            else:
                st.error("One or more checks failed.")
                st.write("**Failure reasons:**")
                for failure in result.failures:
                    st.write(f"- {failure}")

            st.write("**Observed Tools:**")
            if result.observed_tools:
                st.write(", ".join(result.observed_tools))
            else:
                st.write("-")

            if result.answer_preview:
                st.write("**Answer Preview:**")
                st.code(result.answer_preview, language="text")


def run_app() -> None:
    """Render the Streamlit Weekend Wizard interface."""
    st.set_page_config(page_title="Weekend Wizard", page_icon="W", layout="wide")

    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    try:
        readiness = load_readiness()
    except Exception as exc:
        logger.exception("Streamlit readiness check failed: %s", exc)
        st.error(str(exc))
        return

    render_sidebar(readiness)
    chat_tab, observability_tab, prompts_tab, eval_tab = st.tabs(["Chat", "Observability", "Prompts", "Evaluation"])
    with chat_tab:
        render_chat_tab(readiness)
    with observability_tab:
        render_observability(readiness)
    with prompts_tab:
        render_prompts_tab()
    with eval_tab:
        render_eval_tab()

    prompt = st.chat_input("What kind of weekend are you looking for?")
    if prompt:
        st.session_state.pending_prompt = prompt
        st.rerun()


if __name__ == "__main__":
    run_app()
