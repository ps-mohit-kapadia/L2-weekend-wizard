from __future__ import annotations

"""Application root entrypoint for Weekend Wizard."""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from api import run_api
from application.service import WeekendWizardApp
from llm_client import discover_model
from mcp_server import run_mcp_server


def run_streamlit(project_dir: Path, args: Sequence[str] | None = None) -> None:
    """Launch the Streamlit UI from the project root.

    Args:
        project_dir: Project root containing the Streamlit app script.
        args: Optional extra arguments forwarded to ``streamlit run``.

    Raises:
        FileNotFoundError: If the Streamlit app script does not exist.
        SystemExit: If the Streamlit process exits with a non-zero status.
    """
    app_path = project_dir / "streamlit_app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Streamlit app file not found: {app_path}")

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        *(list(args or [])),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


async def run_chat_cli(project_dir: Path, prompt: str, show_observations: bool = False) -> None:
    """Run one Weekend Wizard interaction from the command line.

    Args:
        project_dir: Project root containing the MCP server entrypoint.
        prompt: User prompt to send through the existing agent runtime.
        show_observations: Whether to print collected tool observations after the answer.

    Raises:
        RuntimeError: If model discovery or app startup fails.
    """
    model_name = discover_model(None)
    async with WeekendWizardApp(project_dir / "main.py", model_name, ["mcp-server"]) as app:
        context = app.create_interaction_context()
        result = await app.run_interaction(prompt, context=context)

    print(result.answer)
    if show_observations:
        print()
        print("Tool observations")
        for observation in result.tool_observations:
            print(json.dumps(observation.model_dump(), ensure_ascii=False))


def main(argv: Sequence[str] | None = None) -> None:
    """Dispatch to the supported Weekend Wizard entrypoints.

    Args:
        argv: Optional command-line arguments excluding the script name.

    Raises:
        SystemExit: If an unsupported subcommand is provided.
    """
    args = list(argv if argv is not None else sys.argv[1:])
    project_dir = Path(__file__).resolve().parent

    if args and args[0] == "mcp-server":
        run_mcp_server()
        return

    if args and args[0] == "api":
        run_api()
        return

    if args and args[0] == "streamlit":
        run_streamlit(project_dir, args[1:])
        return

    if args and args[0] == "chat":
        show_observations = "--show-observations" in args[1:]
        prompt_parts = [arg for arg in args[1:] if arg != "--show-observations"]
        if not prompt_parts:
            raise SystemExit("Usage: python main.py chat <prompt> [--show-observations]")
        asyncio.run(run_chat_cli(project_dir, " ".join(prompt_parts), show_observations))
        return

    raise SystemExit("Usage: python main.py [api|streamlit|mcp-server|chat]")


if __name__ == "__main__":
    main()
