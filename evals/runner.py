from __future__ import annotations

"""Run repeatable Weekend Wizard acceptance evals against the HTTP API."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT_SECONDS = 1200


@dataclass(frozen=True)
class EvalCase:
    id: str
    prompt: str
    expected_tools: list[str]
    required_answer_markers: list[str]
    forbidden_tools: list[str]
    forbidden_answer_markers: list[str]
    max_observations: int


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    failures: list[str]
    observed_tools: list[str]
    answer_preview: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Weekend Wizard evals against /chat.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help=f"API base URL. Default: {DEFAULT_API_URL}")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("report.md"))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        cases.append(EvalCase(**payload))
    if not cases:
        raise RuntimeError(f"No eval cases found in {path}")
    return cases


def post_chat(api_url: str, prompt: str, timeout: int) -> dict[str, Any]:
    response = requests.post(f"{api_url.rstrip('/')}/chat", json={"prompt": prompt}, timeout=timeout)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"/chat returned non-JSON response with HTTP {response.status_code}") from exc
    if response.status_code != 200:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        raise RuntimeError(detail or f"/chat returned HTTP {response.status_code}")
    if not isinstance(payload, dict):
        raise RuntimeError("/chat did not return a JSON object")
    return payload


def evaluate_case(case: EvalCase, payload: dict[str, Any]) -> EvalResult:
    answer = str(payload.get("answer") or "")
    observations = payload.get("tool_observations") or []
    observed_tools = [
        str(observation.get("tool_name"))
        for observation in observations
        if isinstance(observation, dict) and observation.get("tool_name")
    ]
    answer_lower = answer.lower()
    failures: list[str] = []

    missing_tools = [tool for tool in case.expected_tools if tool not in observed_tools]
    forbidden_tools = [tool for tool in case.forbidden_tools if tool in observed_tools]
    missing_markers = [
        marker for marker in case.required_answer_markers if marker.lower() not in answer_lower
    ]
    forbidden_markers = [
        marker for marker in case.forbidden_answer_markers if marker.lower() in answer_lower
    ]

    if missing_tools:
        failures.append(f"missing expected tools: {', '.join(missing_tools)}")
    if forbidden_tools:
        failures.append(f"used forbidden tools: {', '.join(forbidden_tools)}")
    if len(observed_tools) > case.max_observations:
        failures.append(f"observation count {len(observed_tools)} exceeded max {case.max_observations}")
    if missing_markers:
        failures.append(f"missing answer markers: {', '.join(missing_markers)}")
    if forbidden_markers:
        failures.append(f"forbidden answer markers present: {', '.join(forbidden_markers)}")

    return EvalResult(
        case_id=case.id,
        passed=not failures,
        failures=failures,
        observed_tools=observed_tools,
        answer_preview=answer.replace("\n", " ")[:240],
    )


def render_report(results: list[EvalResult]) -> str:
    passed = sum(1 for result in results if result.passed)
    lines = [
        "# Weekend Wizard Eval Report",
        "",
        f"Total cases: {len(results)}",
        f"Passed: {passed}",
        f"Failed: {len(results) - passed}",
        "",
        "| Case | Status | Observed Tools | Failures |",
        "| --- | --- | --- | --- |",
    ]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        tools = ", ".join(result.observed_tools) or "-"
        failures = "; ".join(result.failures) or "-"
        lines.append(f"| {result.case_id} | {status} | {tools} | {failures} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    results: list[EvalResult] = []

    for case in cases:
        try:
            payload = post_chat(args.api_url, case.prompt, args.timeout)
            result = evaluate_case(case, payload)
        except Exception as exc:
            result = EvalResult(
                case_id=case.id,
                passed=False,
                failures=[str(exc)],
                observed_tools=[],
                answer_preview="",
            )
        results.append(result)
        print(f"{'PASS' if result.passed else 'FAIL'} {result.case_id}")

    args.report.write_text(render_report(results), encoding="utf-8")
    failed = [result for result in results if not result.passed]
    if failed:
        print(f"Eval failed: {len(failed)} of {len(results)} cases failed. Report: {args.report}")
        raise SystemExit(1)
    print(f"Eval passed: {len(results)} cases. Report: {args.report}")


if __name__ == "__main__":
    main()
