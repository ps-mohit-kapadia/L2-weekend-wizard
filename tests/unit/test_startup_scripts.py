from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from scripts import dev_up


class DevUpScriptTests(unittest.IsolatedAsyncioTestCase):
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
            patch("scripts.dev_up.discover_model", return_value="llama3.1:8b"),
            patch("scripts.dev_up.WeekendWizardApp", return_value=wizard),
            patch("scripts.dev_up.evaluate_runtime_readiness", return_value=readiness),
            redirect_stdout(io.StringIO()) as captured,
        ):
            exit_code = await dev_up._run_preflight()

        self.assertEqual(exit_code, 0)
        self.assertIn("[PASS] Weekend Wizard is ready to run.", captured.getvalue())
        wizard.__aenter__.assert_awaited_once()
        wizard.__aexit__.assert_awaited_once()

    async def test_preflight_reports_operator_friendly_permission_failure(self) -> None:
        wizard = Mock()
        wizard.__aenter__ = AsyncMock(side_effect=RuntimeError("Access is denied"))
        wizard.__aexit__ = AsyncMock()

        with (
            patch("scripts.dev_up.discover_model", return_value="llama3.1:8b"),
            patch("scripts.dev_up.WeekendWizardApp", return_value=wizard),
            redirect_stdout(io.StringIO()) as captured,
        ):
            exit_code = await dev_up._run_preflight()

        self.assertEqual(exit_code, 1)
        self.assertIn("access was denied while starting the MCP runtime", captured.getvalue())
        wizard.__aexit__.assert_not_called()


class DevUpEntrypointTests(unittest.TestCase):
    @patch("scripts.dev_up.subprocess.run")
    @patch("scripts.dev_up.sys.executable", "python")
    @patch("scripts.dev_up._run_preflight_sync", return_value=0)
    def test_dev_up_runs_check_target(self, mock_preflight: Mock, mock_run: Mock) -> None:
        mock_run.return_value.returncode = 0

        dev_up.main(["check"])

        mock_preflight.assert_called_once()
        mock_run.assert_not_called()

    @patch("scripts.dev_up.subprocess.run")
    @patch("scripts.dev_up.sys.executable", "python")
    @patch("scripts.dev_up._run_preflight_sync", return_value=0)
    def test_dev_up_runs_api_target(self, mock_preflight: Mock, mock_run: Mock) -> None:
        mock_run.return_value.returncode = 0

        dev_up.main(["api"])

        mock_preflight.assert_called_once()
        mock_run.assert_called_once_with(
            ["python", "main.py", "api"],
            cwd=dev_up._project_dir(),
            check=False,
        )

    @patch("scripts.dev_up.subprocess.run")
    @patch("scripts.dev_up.sys.executable", "python")
    @patch("scripts.dev_up._run_preflight_sync", return_value=0)
    def test_dev_up_runs_streamlit_target(self, mock_preflight: Mock, mock_run: Mock) -> None:
        mock_run.return_value.returncode = 0

        dev_up.main(["streamlit"])

        mock_preflight.assert_called_once()
        mock_run.assert_called_once_with(
            ["python", "main.py", "streamlit"],
            cwd=dev_up._project_dir(),
            check=False,
        )

    @patch("scripts.dev_up._run_preflight_sync", return_value=1)
    @patch("scripts.dev_up.subprocess.run")
    def test_dev_up_exits_when_preflight_fails(self, mock_run: Mock, _mock_preflight: Mock) -> None:
        with self.assertRaises(SystemExit) as captured:
            dev_up.main(["api"])

        self.assertEqual(captured.exception.code, 1)
        mock_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
