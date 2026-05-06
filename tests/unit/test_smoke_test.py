from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from tests.smoke.smoke_test import (
    start_local_api,
    sys as smoke_sys,
    validate_chat_payload_shape,
    validate_chat_response_status,
)


class SmokeTestContractTests(unittest.TestCase):
    def test_validate_chat_payload_shape_accepts_success_response(self) -> None:
        validate_chat_payload_shape(
            {
                "answer": "A good answer.",
                "tool_observations": [],
                "response_status": "success",
            }
        )

    def test_validate_chat_response_status_rejects_degraded_response(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "marked degraded"):
            validate_chat_response_status(
                {
                    "answer": "Partial fallback answer.",
                    "tool_observations": [],
                    "response_status": "degraded",
                }
            )

    def test_validate_chat_payload_shape_rejects_missing_tool_observations(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "tool_observations list"):
            validate_chat_payload_shape(
                {
                    "answer": "A good answer.",
                    "response_status": "success",
                }
            )

    def test_start_local_api_uses_operator_startup_entrypoint(self) -> None:
        project_dir = Path("C:/project")

        with patch("tests.smoke.smoke_test.subprocess.Popen") as popen_mock:
            start_local_api(project_dir)

        popen_mock.assert_called_once()
        command = popen_mock.call_args.args[0]
        self.assertEqual(command, [smoke_sys.executable, "scripts/dev_up.py", "api"])
        self.assertEqual(popen_mock.call_args.kwargs["cwd"], project_dir)


if __name__ == "__main__":
    unittest.main()
