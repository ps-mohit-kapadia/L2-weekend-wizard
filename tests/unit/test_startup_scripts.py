from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from scripts import check_env, start_api, start_streamlit


class CheckEnvScriptTests(unittest.IsolatedAsyncioTestCase):
    async def test_preflight_reports_readiness_summary_when_runtime_is_ready(self) -> None:
        readiness = Mock(
            status="ready",
            model_name="llama3.1:8b",
            tool_count=6,
            checks=Mock(
                server_path_exists=True,
                ollama_reachable=True,
                model_available=True,
                mcp_session_ready=True,
                tools_discovered=True,
            ),
            details="all good",
        )
        wizard = Mock()
        wizard.__aenter__ = AsyncMock()
        wizard.__aexit__ = AsyncMock()

        with (
            patch("scripts.check_env.discover_model", return_value="llama3.1:8b"),
            patch("scripts.check_env.WeekendWizardApp", return_value=wizard),
            patch("scripts.check_env.evaluate_runtime_readiness", return_value=readiness),
            redirect_stdout(io.StringIO()) as captured,
        ):
            exit_code = await check_env._run_preflight()

        self.assertEqual(exit_code, 0)
        self.assertIn("[PASS] Weekend Wizard is ready to run.", captured.getvalue())
        wizard.__aenter__.assert_awaited_once()
        wizard.__aexit__.assert_awaited_once()

    async def test_preflight_reports_operator_friendly_permission_failure(self) -> None:
        wizard = Mock()
        wizard.__aenter__ = AsyncMock(side_effect=RuntimeError("Access is denied"))
        wizard.__aexit__ = AsyncMock()

        with (
            patch("scripts.check_env.discover_model", return_value="llama3.1:8b"),
            patch("scripts.check_env.WeekendWizardApp", return_value=wizard),
            redirect_stdout(io.StringIO()) as captured,
        ):
            exit_code = await check_env._run_preflight()

        self.assertEqual(exit_code, 1)
        self.assertIn("access was denied while starting the MCP runtime", captured.getvalue())
        wizard.__aexit__.assert_not_called()


class StartupScriptTests(unittest.TestCase):
    @patch("scripts.start_api.subprocess.run")
    @patch("scripts.start_api.sys.executable", "python")
    def test_start_api_runs_preflight_from_project_root(self, mock_run: Mock) -> None:
        mock_run.return_value.returncode = 0

        start_api._run_preflight(Path("C:/repo"))

        mock_run.assert_called_once_with(
            ["python", str(Path("C:/repo") / "scripts" / "check_env.py")],
            cwd=Path("C:/repo"),
            check=False,
        )

    @patch("scripts.start_streamlit.subprocess.run")
    @patch("scripts.start_streamlit.sys.executable", "python")
    def test_start_streamlit_runs_preflight_from_project_root(self, mock_run: Mock) -> None:
        mock_run.return_value.returncode = 0

        start_streamlit._run_preflight(Path("C:/repo"))

        mock_run.assert_called_once_with(
            ["python", str(Path("C:/repo") / "scripts" / "check_env.py")],
            cwd=Path("C:/repo"),
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
