from __future__ import annotations

import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from evaluations.run_evaluations import EvaluationCase, EvaluationResult, load_cases, print_summary, score_case


class EvaluationTests(unittest.TestCase):
    def test_load_cases_reads_json_dataset(self) -> None:
        path = Path("C:/repo/evaluations/cases.json")
        with patch.object(
            Path,
            "read_text",
            return_value="""
            [
              {
                "id": "joke-only",
                "category": "single_tool",
                "prompt": "Tell me a joke.",
                "required_tools": ["random_joke"],
                "forbidden_tools": ["trivia"],
                "min_observations": 1,
                "expect_answer": true
              }
            ]
            """.strip(),
        ):
            cases = load_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].case_id, "joke-only")
        self.assertEqual(cases[0].required_tools, ["random_joke"])
        self.assertFalse(cases[0].allow_degraded)

    def test_score_case_passes_when_contract_is_satisfied(self) -> None:
        case = EvaluationCase(
            case_id="joke-only",
            category="single_tool",
            prompt="Tell me a joke.",
            required_tools=["random_joke"],
            forbidden_tools=["trivia"],
            min_observations=1,
        )
        payload = {
            "answer": "Here's a joke.",
            "tool_observations": [
                {
                    "tool_name": "random_joke",
                    "args": {},
                    "payload": '{"joke": "hello"}',
                }
            ],
            "response_status": "success",
        }

        result = score_case(case, payload)

        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, [])
        self.assertEqual(result.tool_names, ["random_joke"])

    def test_score_case_reports_missing_and_forbidden_tools(self) -> None:
        case = EvaluationCase(
            case_id="weather-dog",
            category="multi_tool",
            prompt="Weather and dog please.",
            required_tools=["get_weather", "random_dog"],
            forbidden_tools=["trivia"],
            min_observations=2,
        )
        payload = {
            "answer": "Done.",
            "tool_observations": [
                {"tool_name": "get_weather", "args": {}, "payload": "{}"},
                {"tool_name": "trivia", "args": {}, "payload": "{}"},
            ],
            "response_status": "success",
        }

        result = score_case(case, payload)

        self.assertFalse(result.passed)
        self.assertTrue(any("Missing required tools: random_dog." == reason for reason in result.reasons))
        self.assertTrue(any("Observed forbidden tools: trivia." == reason for reason in result.reasons))

    def test_score_case_reports_empty_answer_and_missing_observations(self) -> None:
        case = EvaluationCase(
            case_id="weather-city",
            category="weather",
            prompt="What's the weather?",
            required_tools=["get_weather"],
            min_observations=1,
            expect_answer=True,
        )
        payload = {
            "answer": "   ",
            "tool_observations": [],
            "response_status": "success",
        }

        result = score_case(case, payload)

        self.assertFalse(result.passed)
        self.assertTrue(any("non-empty answer" in reason for reason in result.reasons))
        self.assertTrue(any("Expected at least 1 observations" in reason for reason in result.reasons))

    def test_score_case_fails_degraded_response_by_default(self) -> None:
        case = EvaluationCase(
            case_id="books-only",
            category="books",
            prompt="Plan a weekend in Las Vegas with 3 adventure books.",
            required_tools=["book_recs"],
            min_observations=1,
        )
        payload = {
            "answer": "Here are 3 adventure books.",
            "tool_observations": [
                {"tool_name": "book_recs", "args": {"topic": "adventure", "limit": 3}, "payload": "{}"},
            ],
            "response_status": "degraded",
        }

        result = score_case(case, payload)

        self.assertFalse(result.passed)
        self.assertTrue(any("marked degraded" in reason for reason in result.reasons))

    def test_score_case_reports_required_tool_failures_for_degraded_response(self) -> None:
        case = EvaluationCase(
            case_id="weather-joke",
            category="multi_tool",
            prompt="Give me the weather and a joke.",
            required_tools=["get_weather", "random_joke"],
            min_observations=2,
        )
        payload = {
            "answer": "I could only complete part of that request.",
            "tool_observations": [
                {
                    "tool_name": "get_weather",
                    "args": {"latitude": 40.7128, "longitude": -74.0060},
                    "payload": '{"error":"get_weather failed","details":"timeout"}',
                },
                {
                    "tool_name": "random_joke",
                    "args": {},
                    "payload": '{"joke":"hello"}',
                },
            ],
            "response_status": "degraded",
        }

        result = score_case(case, payload)

        self.assertFalse(result.passed)
        self.assertTrue(any("Required tools failed during execution: get_weather." == reason for reason in result.reasons))

    def test_score_case_allows_degraded_response_when_case_opts_in(self) -> None:
        case = EvaluationCase(
            case_id="reflection-only-fallback",
            category="books",
            prompt="Give me 3 adventure books.",
            required_tools=["book_recs"],
            min_observations=1,
            allow_degraded=True,
        )
        payload = {
            "answer": "Here are 3 adventure books.",
            "tool_observations": [
                {"tool_name": "book_recs", "args": {"topic": "adventure", "limit": 3}, "payload": "{}"},
            ],
            "response_status": "degraded",
        }

        result = score_case(case, payload)

        self.assertTrue(result.passed)
        self.assertEqual(result.reasons, [])

    @patch("evaluations.run_evaluations.requests.post", side_effect=requests.Timeout("too slow"))
    def test_evaluate_case_returns_failed_result_on_timeout(self, _mock_post: Mock) -> None:
        from evaluations.run_evaluations import evaluate_case

        case = EvaluationCase(
            case_id="weekend-plan",
            category="weekend_plan",
            prompt="Plan me a weekend.",
        )

        result = evaluate_case(
            "http://127.0.0.1:8000",
            {"X-API-Key": "secret-key"},
            case,
            request_timeout=180,
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.tool_names, [])
        self.assertTrue(any("timed out" in reason for reason in result.reasons))

    def test_print_summary_hides_timing_by_default(self) -> None:
        results = [
            EvaluationResult(
                case_id="joke-only",
                passed=True,
                reasons=[],
                tool_names=["random_joke"],
                observation_count=1,
                answer_length=42,
                duration_seconds=12.3,
            )
        ]

        with patch("sys.stdout", new=StringIO()) as captured:
            print_summary(results, show_timing=False)

        output = captured.getvalue()
        self.assertIn("passed: 1/1", output)
        self.assertNotIn("duration=", output)
        self.assertNotIn("total_duration", output)

    def test_print_summary_shows_timing_when_enabled(self) -> None:
        results = [
            EvaluationResult(
                case_id="joke-only",
                passed=True,
                reasons=[],
                tool_names=["random_joke"],
                observation_count=1,
                answer_length=42,
                duration_seconds=12.3,
            ),
            EvaluationResult(
                case_id="dog-only",
                passed=False,
                reasons=["Request timed out after 180s."],
                tool_names=[],
                observation_count=0,
                answer_length=0,
                duration_seconds=180.0,
            ),
        ]

        with patch("sys.stdout", new=StringIO()) as captured:
            print_summary(results, show_timing=True)

        output = captured.getvalue()
        self.assertIn("total_duration", output)
        self.assertIn("slowest_case: dog-only", output)
        self.assertIn("duration=12.3s", output)


if __name__ == "__main__":
    unittest.main()
