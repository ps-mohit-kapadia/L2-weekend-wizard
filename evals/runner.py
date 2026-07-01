from __future__ import annotations

"""Run repeatable Weekend Wizard acceptance evals against the HTTP API."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

# Add repository root to Python path for config import
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from config.config import get_settings


def build_chat_headers() -> dict[str, str]:
    """Return optional headers required by the backend chat endpoint."""
    api_key = get_settings().api_key
    if api_key is None:
        return {}
    return {"X-API-Key": api_key}


DEFAULT_TIMEOUT_SECONDS = 1200


@dataclass(frozen=True)
class EvalCase:
    """One acceptance eval case loaded from JSONL.

    Attributes:
        id: Stable case identifier used in reports.
        prompt: User prompt sent to the Weekend Wizard `/chat` endpoint.
        expected_tools: Tool names that must appear in tool observations.
        required_answer_markers: Text markers that must appear in the answer.
        forbidden_tools: Tool names that must not appear in tool observations.
        forbidden_answer_markers: Text markers that must not appear in the answer.
        max_observations: Maximum allowed number of tool observations.
    """

    id: str
    prompt: str
    expected_tools: list[str]
    required_answer_markers: list[str]
    forbidden_tools: list[str]
    forbidden_answer_markers: list[str]
    max_observations: int


@dataclass(frozen=True)
class EvalResult:
    """Result for one evaluated acceptance case.

    Attributes:
        case_id: Stable identifier of the evaluated case.
        passed: Whether the case satisfied all rubric checks.
        failures: Human-readable failure reasons.
        observed_tools: Tool names observed in the `/chat` response.
        answer_preview: Short answer excerpt for diagnostics.
    """

    case_id: str
    passed: bool
    failures: list[str]
    observed_tools: list[str]
    answer_preview: str


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the eval runner.

    Returns:
        Parsed command-line namespace.
    """

    parser = argparse.ArgumentParser(description="Run Weekend Wizard evals against /chat.")
    parser.add_argument("--api-url", default=None, help="API base URL. Default: from config.api_url")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.jsonl"))
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("report.md"))
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def load_cases(path: Path) -> list[EvalCase]:
    """Load eval cases from a JSONL file.

    Args:
        path: Path to the JSONL case file.

    Returns:
        Eval cases in file order.

    Raises:
        RuntimeError: If no cases are found.
        json.JSONDecodeError: If a case line is not valid JSON.
        TypeError: If a case payload does not match `EvalCase`.
    """

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
    """Send one prompt to the Weekend Wizard `/chat` endpoint.

    Args:
        api_url: Base URL for the Weekend Wizard API.
        prompt: User prompt for the eval case.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response payload.

    Raises:
        RuntimeError: If the API response is non-JSON, non-200, or not an object.
        requests.RequestException: If the HTTP request fails.
    """

    headers = build_chat_headers()
    response = requests.post(
        f"{api_url.rstrip('/')}/chat",
        json={"prompt": prompt},
        headers=headers,
        timeout=timeout
    )
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
    """Evaluate one `/chat` response against one case contract.

    Args:
        case: Eval case containing expected and forbidden behavior.
        payload: Parsed `/chat` response payload.

    Returns:
        Case result with pass/fail status and diagnostics.
    """

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
    """Render eval results as a Markdown report.

    Args:
        results: Eval results to summarize.

    Returns:
        Markdown report text.
    """

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
    """Run all configured eval cases and write the Markdown report."""

    args = parse_args()
    api_url = args.api_url or get_settings().api_url
    cases = load_cases(args.cases)
    results: list[EvalResult] = []

    for case in cases:
        try:
            payload = post_chat(api_url, case.prompt, args.timeout)
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
