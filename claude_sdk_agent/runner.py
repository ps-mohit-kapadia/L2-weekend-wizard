from __future__ import annotations

"""Runner for the first Claude Agent SDK Weekend Wizard slice."""

from typing import Any

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage, TextBlock

from claude_sdk_agent.config import ClaudeSdkAgentConfig, get_default_config
from claude_sdk_agent.prompts import build_system_prompt
from claude_sdk_agent.tools import build_sdk_server
from logger.tracing.request_trace import create_correlation_id


def build_agent_options(config: ClaudeSdkAgentConfig) -> ClaudeAgentOptions:
    """Build deterministic Claude Agent SDK options for the current slice."""
    sdk_server = build_sdk_server(config)
    return ClaudeAgentOptions(
        system_prompt=build_system_prompt(),
        mcp_servers={config.server_key: sdk_server},
        allowed_tools=config.allowed_tool_names,
        max_turns=config.max_turns,
        model=config.model,
        cwd=config.cwd,
    )


def build_dry_run_summary(config: ClaudeSdkAgentConfig) -> dict[str, Any]:
    """Return a non-network summary of the configured SDK agent slice."""
    system_prompt = build_system_prompt()
    return {
        "mode": "no-live-call",
        "server_key": config.server_key,
        "server_name": config.server_name,
        "model": config.model,
        "allowed_tools": config.allowed_tool_names,
        "max_turns": config.max_turns,
        "cwd": str(config.cwd),
        "system_prompt": system_prompt,
    }


def _extract_assistant_text(message: AssistantMessage) -> str:
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(block.text)
    return "\n".join(part for part in parts if part.strip())


async def run_claude_sdk_prompt(
    prompt: str,
    *,
    config: ClaudeSdkAgentConfig | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one prompt through the Claude Agent SDK slice.

    When ``dry_run`` is true, no remote call is attempted and a deterministic
    summary of the configured path is returned instead.
    """
    resolved = config or get_default_config()
    correlation_id = create_correlation_id()
    if dry_run:
        return {
            "mode": "dry-run",
            "correlation_id": correlation_id,
            "prompt": prompt,
            "config": build_dry_run_summary(resolved),
        }

    options = build_agent_options(resolved)
    final_text = ""
    final_result = None

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                text = _extract_assistant_text(message)
                if text.strip():
                    final_text = text.strip()
            elif isinstance(message, ResultMessage):
                final_result = message.result or final_result

    return {
        "mode": "live",
        "correlation_id": correlation_id,
        "answer": final_text or final_result or "",
    }
