from __future__ import annotations

"""Deterministic configuration for the Claude Agent SDK Weekend Wizard slice."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClaudeSdkAgentConfig:
    """Static configuration for the first Claude Agent SDK slice."""

    server_key: str
    server_name: str
    server_version: str
    max_turns: int
    model: str | None
    cwd: Path

    @property
    def joke_tool_name(self) -> str:
        """Return the fully-qualified Claude SDK MCP tool name for the joke tool."""
        return f"mcp__{self.server_key}__random_joke"


def get_default_config() -> ClaudeSdkAgentConfig:
    """Return the deterministic default SDK configuration for this repo."""
    project_root = Path(__file__).resolve().parent.parent
    return ClaudeSdkAgentConfig(
        server_key="weekend_wizard",
        server_name="weekend-wizard-sdk",
        server_version="0.1.0",
        max_turns=4,
        model=None,
        cwd=project_root,
    )
