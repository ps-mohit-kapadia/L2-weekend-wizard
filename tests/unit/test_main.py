from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import main


class MainEntrypointTests(unittest.TestCase):
    def test_main_requires_an_explicit_supported_subcommand(self) -> None:
        with self.assertRaises(SystemExit) as captured:
            main.main([])

        self.assertIn("Usage:", str(captured.exception))

    @patch("main.run_mcp_server")
    def test_main_dispatches_to_mcp_server_subcommand(self, mock_run_mcp_server: Mock) -> None:
        main.main(["mcp-server"])

        mock_run_mcp_server.assert_called_once()

    @patch("main.run_api")
    def test_main_dispatches_to_api_subcommand(self, mock_run_api: Mock) -> None:
        main.main(["api"])

        mock_run_api.assert_called_once()

    @patch("main.subprocess.run")
    @patch("main.sys.executable", "python")
    @patch("pathlib.Path.exists", return_value=True)
    def test_main_dispatches_to_streamlit_subcommand(
        self,
        _mock_exists: Mock,
        mock_run: Mock,
    ) -> None:
        mock_run.return_value.returncode = 0

        main.main(["streamlit", "--server.headless=true"])

        mock_run.assert_called_once_with(
            [
                "python",
                "-m",
                "streamlit",
                "run",
                str(Path(main.__file__).resolve().parent / "streamlit_app.py"),
                "--server.headless=true",
            ],
            check=False,
        )

    def test_main_rejects_unknown_subcommands(self) -> None:
        with self.assertRaises(SystemExit) as captured:
            main.main(["cli"])

        self.assertIn("Usage:", str(captured.exception))

    def test_run_streamlit_raises_when_app_script_is_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            main.run_streamlit(Path("C:/missing/project"))


if __name__ == "__main__":
    unittest.main()
