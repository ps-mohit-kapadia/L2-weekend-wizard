from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from mcp_runtime.client import McpService, ToolInvocationError


class McpServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_call_tool_raises_before_initialization(self) -> None:
        service = McpService(Path("mcp_server.py"))

        with self.assertRaises(RuntimeError):
            await service.call_tool("get_weather", {"latitude": 1.0, "longitude": 2.0})

    async def test_tool_names_reflect_loaded_tools(self) -> None:
        service = McpService(Path("mcp_server.py"))
        service._tools = [SimpleNamespace(name="get_weather"), SimpleNamespace(name="random_joke")]

        self.assertEqual(service.tool_names, ["get_weather", "random_joke"])

    async def test_server_args_are_retained_for_spawn(self) -> None:
        service = McpService(Path("main.py"), server_args=["mcp-server"])

        self.assertEqual(service._server_args, ["mcp-server"])

    async def test_call_tool_wraps_operational_transport_failures(self) -> None:
        service = McpService(Path("mcp_server.py"))
        service._session = SimpleNamespace(call_tool=AsyncMock(side_effect=OSError("broken pipe")))

        with self.assertRaises(ToolInvocationError):
            await service.call_tool("get_weather", {"latitude": 1.0, "longitude": 2.0})

    async def test_call_tool_serializes_shared_session_access(self) -> None:
        service = McpService(Path("mcp_server.py"))
        active_calls = 0
        max_active_calls = 0

        async def fake_call_tool(_tool_name: str, _args: dict[str, float]) -> dict[str, str]:
            nonlocal active_calls, max_active_calls
            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)
            await asyncio.sleep(0)
            active_calls -= 1
            return {"status": "ok"}

        service._session = SimpleNamespace(call_tool=fake_call_tool)

        await asyncio.gather(
            service.call_tool("get_weather", {"latitude": 1.0, "longitude": 2.0}),
            service.call_tool("random_joke", {}),
        )

        self.assertEqual(max_active_calls, 1)


if __name__ == "__main__":
    unittest.main()
