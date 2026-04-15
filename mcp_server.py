from __future__ import annotations

"""Bootstrap the MCP tool server for Weekend Wizard."""

from mcp_runtime.registry import mcp
from tools import books, entertainment, geo, weather  # noqa: F401


def run_mcp_server() -> None:
    """Start the MCP tool server for stdio-based clients."""
    mcp.run()


if __name__ == "__main__":
    run_mcp_server()
