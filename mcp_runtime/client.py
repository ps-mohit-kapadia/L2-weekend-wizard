from __future__ import annotations

"""MCP client layer for server lifecycle, tool discovery, and tool calls."""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Protocol

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError


def extract_tool_payload(result: Any) -> Any:
    """Extract one logical payload object from a raw MCP tool result."""
    if getattr(result, "content", None):
        for item in result.content:
            text = getattr(item, "text", None)
            if text is not None:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            if hasattr(item, "model_dump"):
                return item.model_dump()
            if hasattr(item, "model_dump_json"):
                dumped = item.model_dump_json()
                try:
                    return json.loads(dumped)
                except json.JSONDecodeError:
                    return dumped
            return str(item)
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "model_dump_json"):
        dumped = result.model_dump_json()
        try:
            return json.loads(dumped)
        except json.JSONDecodeError:
            return dumped
    return str(result)


class ToolGateway(Protocol):
    """Protocol for components that can invoke MCP tools.

    Methods:
        call_tool: Invoke a named tool with JSON-compatible arguments.
    """

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Invoke a named tool with JSON-compatible arguments."""


class ToolInvocationError(RuntimeError):
    """Operational failure while invoking a tool through the MCP client."""


class McpService:
    """Manage the MCP stdio server lifecycle and client session.

    Args:
        server_path: Absolute path to the MCP server entrypoint.
        command: Python executable used to launch the MCP server.
        server_args: Additional arguments passed to the server entrypoint.
    """

    def __init__(
        self,
        server_path: Path,
        command: str | None = None,
        server_args: List[str] | None = None,
    ) -> None:
        self._server_path = server_path
        self._command = command or sys.executable
        self._server_args = list(server_args or [])
        self._exit_stack: AsyncExitStack | None = None
        self._call_lock = asyncio.Lock()
        self._session: ClientSession | None = None
        self._tools: List[Any] = []

    @property
    def tools(self) -> tuple[Any, ...]:
        """Return MCP tool descriptors discovered during initialization.

        Returns:
            An immutable snapshot of the discovered MCP tool descriptors.
        """
        return tuple(self._tools)

    @property
    def tool_names(self) -> tuple[str, ...]:
        """Return discovered tool names.

        Returns:
            An immutable snapshot of the tool names exposed by the MCP server.
        """
        return tuple(tool.name for tool in self._tools)

    async def __aenter__(self) -> McpService:
        """Start the MCP server and initialize the client session.

        Returns:
            The initialized MCP service instance.
        """
        self._exit_stack = AsyncExitStack()
        stdio_transport = await self._exit_stack.enter_async_context(
            stdio_client(
                StdioServerParameters(
                    command=self._command,
                    args=[str(self._server_path), *self._server_args],
                )
            )
        )
        reader, writer = stdio_transport

        self._session = await self._exit_stack.enter_async_context(ClientSession(reader, writer))
        await self._session.initialize()
        self._tools = (await self._session.list_tools()).tools
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, exc_tb: Any) -> None:
        """Tear down the MCP client session and child server process."""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._session = None
        self._tools = []

    async def call_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Invoke an MCP tool through the managed client session.

        Args:
            tool_name: Name of the MCP tool to invoke.
            args: JSON-compatible arguments for the tool call.

        Returns:
            The raw MCP tool result.

        Raises:
            RuntimeError: If the service has not been initialized yet.
            ToolInvocationError: If the MCP transport or server reports a tool-call failure.
        """
        if self._session is None:
            raise RuntimeError("MCP service has not been initialized.")
        try:
            async with self._call_lock:
                return await self._session.call_tool(tool_name, args)
        except (McpError, OSError, TimeoutError) as exc:
            raise ToolInvocationError(str(exc)) from exc
