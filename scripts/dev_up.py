from __future__ import annotations

"""Operator-facing startup entrypoint for Weekend Wizard."""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from api import evaluate_runtime_readiness
from application.service import WeekendWizardApp
from llm_client import discover_model


def _project_dir() -> Path:
    """Return the project root for the current script location."""
    return PROJECT_DIR


def _is_access_denied_error(exc: BaseException) -> bool:
    """Return whether an exception looks like a permission-denied startup failure.

    Args:
        exc: Exception raised while trying to initialize the runtime.

    Returns:
        True when the exception or its message indicates access was denied.
    """
    if isinstance(exc, PermissionError):
        return True

    message = str(exc).lower()
    return "access is denied" in message or "permission denied" in message


async def _run_preflight() -> int:
    """Run runtime preflight checks and print a concise summary.

    Returns:
        Process exit code for the preflight result.
    """
    server_path = PROJECT_DIR / "main.py"

    try:
        model_name = discover_model(None)
    except Exception as exc:
        print(f"[FAIL] Model preflight failed: {exc}")
        return 1

    wizard = WeekendWizardApp(server_path, model_name, ["mcp-server"])
    try:
        await wizard.__aenter__()
    except Exception as exc:
        if _is_access_denied_error(exc):
            print(
                "[FAIL] Runtime startup preflight failed: access was denied while starting the MCP runtime. "
                "If you are running in a restricted shell or sandbox, run this check from a normal local terminal."
            )
            return 1
        print(f"[FAIL] Runtime startup preflight failed: {exc}")
        return 1

    try:
        readiness = evaluate_runtime_readiness(wizard)
    finally:
        await wizard.__aexit__(None, None, None)

    print("Weekend Wizard preflight")
    print(f"- status: {readiness.status}")
    print(f"- model: {readiness.model_name}")
    print(f"- tool_count: {readiness.tool_count}")
    print(f"- server_path_exists: {readiness.checks.server_path_exists}")
    print(f"- ollama_reachable: {readiness.checks.ollama_reachable}")
    print(f"- model_available: {readiness.checks.model_available}")
    print(f"- mcp_session_ready: {readiness.checks.mcp_session_ready}")
    print(f"- tools_discovered: {readiness.checks.tools_discovered}")
    if readiness.details:
        print(f"- details: {readiness.details}")

    if readiness.status != "ready":
        print("[FAIL] Weekend Wizard is not ready to start.")
        return 1

    print("[PASS] Weekend Wizard is ready to run.")
    return 0


def _run_preflight_sync() -> int:
    """Run the asynchronous preflight checks from the CLI entrypoint."""
    return asyncio.run(_run_preflight())


def _run_service(project_dir: Path, target: str) -> None:
    """Run a long-lived local service after preflight succeeds.

    Args:
        project_dir: Project root used as the subprocess working directory.
        target: Service subcommand to dispatch through ``main.py``.

    Raises:
        SystemExit: If the service exits with a non-zero status.
    """
    command = [sys.executable, "main.py", target]
    completed = subprocess.run(command, cwd=project_dir, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for local developer startup commands."""
    parser = argparse.ArgumentParser(
        prog="python scripts/dev_up.py",
        description="Run a preflight check or start a Weekend Wizard local service.",
    )
    parser.add_argument(
        "target",
        choices=("check", "api", "streamlit"),
        help="Target action to run after resolving the local project environment.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the requested local startup action.

    Args:
        argv: Optional command-line arguments excluding the script name.
    """
    args = _build_parser().parse_args(argv)
    project_dir = _project_dir()

    if args.target == "check":
        exit_code = _run_preflight_sync()
        if exit_code != 0:
            raise SystemExit(exit_code)
        return

    preflight_exit_code = _run_preflight_sync()
    if preflight_exit_code != 0:
        raise SystemExit(preflight_exit_code)

    if args.target == "api":
        _run_service(project_dir, "api")
        return

    _run_service(project_dir, "streamlit")


if __name__ == "__main__":
    main()
