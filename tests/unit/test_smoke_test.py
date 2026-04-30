from __future__ import annotations

import unittest

from tests.smoke.smoke_test import validate_chat_payload


class SmokeTestContractTests(unittest.TestCase):
    def test_validate_chat_payload_accepts_success_response(self) -> None:
        validate_chat_payload(
            {
                "answer": "A good answer.",
                "tool_observations": [],
                "response_status": "success",
            }
        )

    def test_validate_chat_payload_rejects_degraded_response(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "marked degraded"):
            validate_chat_payload(
                {
                    "answer": "Partial fallback answer.",
                    "tool_observations": [],
                    "response_status": "degraded",
                }
            )


if __name__ == "__main__":
    unittest.main()
