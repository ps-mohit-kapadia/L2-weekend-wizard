from __future__ import annotations

import unittest
from unittest.mock import patch

from tests.smoke import smoke_test


class SmokeTestTimeoutConfigTests(unittest.TestCase):
    def test_resolve_chat_timeout_uses_default_when_env_missing(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(
                smoke_test.resolve_chat_timeout_seconds(),
                smoke_test.DEFAULT_CHAT_TIMEOUT_SECONDS,
            )

    def test_resolve_chat_timeout_uses_env_override_when_present(self) -> None:
        with patch.dict("os.environ", {"WEEKEND_WIZARD_REQUEST_TIMEOUT": "600"}, clear=True):
            self.assertEqual(smoke_test.resolve_chat_timeout_seconds(), 600)

    def test_resolve_chat_timeout_raises_clear_error_for_invalid_env(self) -> None:
        with patch.dict("os.environ", {"WEEKEND_WIZARD_REQUEST_TIMEOUT": "slow"}, clear=True):
            with self.assertRaises(RuntimeError) as captured:
                smoke_test.resolve_chat_timeout_seconds()

        self.assertIn("WEEKEND_WIZARD_REQUEST_TIMEOUT", str(captured.exception))
        self.assertIn("valid integer", str(captured.exception))


if __name__ == "__main__":
    unittest.main()
