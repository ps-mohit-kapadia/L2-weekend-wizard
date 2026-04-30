from __future__ import annotations

import unittest

from tests.smoke.smoke_test import (
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


if __name__ == "__main__":
    unittest.main()
