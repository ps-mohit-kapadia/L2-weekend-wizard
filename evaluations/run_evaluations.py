from __future__ import annotations

"""Run lightweight contract evaluations against the Weekend Wizard API."""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from config.config import get_settings
API_KEY_HEADER = "X-API-Key"


@dataclass(slots=True)
class EvaluationCase:
    """One prompt and its contract expectations.

    Attributes:
        case_id: Stable identifier for the evaluation case.
        category: Logical grouping for the prompt.
        prompt: Prompt sent to the API.
        required_tools: Tool names that must appear in observations.
        forbidden_tools: Tool names that must not appear in observations.
        min_observations: Minimum number of observations expected.
        expect_answer: Whether the API should return a non-empty answer.
    """

    case_id: str
    category: str
    prompt: str
    required_tools: list[str] = field(default_factory=list)
    forbidden_tools: list[str] = field(default_factory=list)
    min_observations: int = 0
    expect_answer: bool = True


@dataclass(slots=True)
class EvaluationResult:
    """Scored result for one evaluation case.

    Attributes:
        case_id: Stable identifier for the evaluation case.
        passed: Whether the response satisfied the case contract.
        reasons: Contract failures encountered while scoring.
        tool_names: Tool names observed in the API response.
        observation_count: Number of tool observations returned by the API.
        answer_length: Length of the final answer text.
        duration_seconds: End-to-end case duration in seconds.
    """

    case_id: str
    passed: bool
    reasons: list[str]
    tool_names: list[str]
    observation_count: int
    answer_length: int
    duration_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation runner."""
    parser = argparse.ArgumentParser(
        description="Run repeatable contract evaluations against the Weekend Wizard API."
    )
    parser.add_argument(
        "--api-url",
        default=get_settings().api_url,
        help="Base URL of the FastAPI service. Default: configured WEEKEND_WIZARD_API_URL/runtime host+port.",
    )
    parser.add_argument(
        "--cases-path",
        default=str(Path(__file__).resolve().with_name("cases.json")),
        help="Path to the JSON file containing evaluation cases.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=90,
        help="Seconds to wait for the API to become ready. Default: 90",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Seconds between readiness checks. Default: 2.0",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=180,
        help="Seconds to wait for each /chat response. Default: 180",
    )
    parser.add_argument(
        "--no-start-api",
        action="store_true",
        help="Fail instead of starting a local API process when the API is not already reachable.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop evaluation after the first failing case.",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Include per-case and aggregate timing details in the evaluation summary.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load evaluation cases from JSON.

    Args:
        path: JSON file containing the evaluation dataset.

    Returns:
        Parsed evaluation cases.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvaluationCase(
            case_id=item["id"],
            category=item["category"],
            prompt=item["prompt"],
            required_tools=list(item.get("required_tools", [])),
            forbidden_tools=list(item.get("forbidden_tools", [])),
            min_observations=int(item.get("min_observations", 0)),
            expect_answer=bool(item.get("expect_answer", True)),
        )
        for item in payload
    ]


def get_api_headers() -> dict[str, str]:
    """Return required headers for protected API requests."""
    api_key = os.getenv("WEEKEND_WIZARD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("WEEKEND_WIZARD_API_KEY is not configured for evaluations.")
    return {API_KEY_HEADER: api_key}


def api_is_reachable(base_url: str, headers: dict[str, str], timeout: int = 3) -> bool:
    """Return whether the API health endpoint is reachable."""
    try:
        response = requests.get(f"{base_url}/health", headers=headers, timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


def read_process_output(process: subprocess.Popen[str]) -> str:
    """Return captured process output when available."""
    if process.stdout is None:
        return ""
    return process.stdout.read().strip()


def wait_for_ready(
    base_url: str,
    headers: dict[str, str],
    timeout_seconds: int,
    poll_interval: float,
    process: subprocess.Popen[str] | None = None,
) -> dict[str, Any]:
    """Wait until the API reports ready or raise with the last known details."""
    deadline = time.monotonic() + timeout_seconds
    last_details = "API did not become ready."

    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            output = read_process_output(process)
            raise RuntimeError(
                "Weekend Wizard API process exited before becoming ready."
                + (f"\n\nProcess output:\n{output}" if output else "")
            )

        try:
            response = requests.get(f"{base_url}/ready", headers=headers, timeout=10)
            payload = response.json()
        except requests.RequestException as exc:
            last_details = str(exc)
            time.sleep(poll_interval)
            continue
        except ValueError as exc:
            raise RuntimeError("Weekend Wizard API returned invalid JSON from /ready.") from exc

        if response.status_code == 200 and payload.get("status") == "ready":
            return payload

        last_details = payload.get("details") or json.dumps(payload)
        time.sleep(poll_interval)

    raise RuntimeError(f"Weekend Wizard API did not become ready within {timeout_seconds}s: {last_details}")


def start_local_api(project_dir: Path) -> subprocess.Popen[str]:
    """Start a local Weekend Wizard API process through the operator entrypoint."""
    command = [sys.executable, "scripts/dev_up.py", "api"]
    return subprocess.Popen(
        command,
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_process(process: subprocess.Popen[str]) -> None:
    """Terminate a spawned process best-effort."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def score_case(case: EvaluationCase, payload: dict[str, Any]) -> EvaluationResult:
    """Score one API response against an evaluation contract.

    Args:
        case: Evaluation case describing required behavior.
        payload: JSON payload returned by ``/chat``.

    Returns:
        Scored evaluation result for the case.
    """
    reasons: list[str] = []
    answer = payload.get("answer")
    observations = payload.get("tool_observations")

    if case.expect_answer and (not isinstance(answer, str) or not answer.strip()):
        reasons.append("Response did not include a non-empty answer.")

    if not isinstance(observations, list):
        reasons.append("Response did not include a tool_observations list.")
        observations = []

    tool_names = [
        item.get("tool_name")
        for item in observations
        if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
    ]

    if len(observations) < case.min_observations:
        reasons.append(
            f"Expected at least {case.min_observations} observations but saw {len(observations)}."
        )

    missing_tools = sorted(set(case.required_tools) - set(tool_names))
    if missing_tools:
        reasons.append(f"Missing required tools: {', '.join(missing_tools)}.")

    present_forbidden = sorted(set(case.forbidden_tools).intersection(tool_names))
    if present_forbidden:
        reasons.append(f"Observed forbidden tools: {', '.join(present_forbidden)}.")

    return EvaluationResult(
        case_id=case.case_id,
        passed=not reasons,
        reasons=reasons,
        tool_names=tool_names,
        observation_count=len(observations),
        answer_length=len(answer) if isinstance(answer, str) else 0,
        duration_seconds=0.0,
    )


def evaluate_case(
    base_url: str,
    headers: dict[str, str],
    case: EvaluationCase,
    request_timeout: int,
) -> EvaluationResult:
    """Send one evaluation prompt through the API and score the response."""
    started_at = time.monotonic()

    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"prompt": case.prompt},
            headers=headers,
            timeout=request_timeout,
        )
    except requests.Timeout:
        return EvaluationResult(
            case_id=case.case_id,
            passed=False,
            reasons=[f"Request timed out after {request_timeout}s."],
            tool_names=[],
            observation_count=0,
            answer_length=0,
            duration_seconds=time.monotonic() - started_at,
        )
    except requests.RequestException as exc:
        return EvaluationResult(
            case_id=case.case_id,
            passed=False,
            reasons=[f"Request failed: {exc}"],
            tool_names=[],
            observation_count=0,
            answer_length=0,
            duration_seconds=time.monotonic() - started_at,
        )

    try:
        payload = response.json()
    except ValueError:
        return EvaluationResult(
            case_id=case.case_id,
            passed=False,
            reasons=["API returned invalid JSON from /chat."],
            tool_names=[],
            observation_count=0,
            answer_length=0,
            duration_seconds=time.monotonic() - started_at,
        )

    if response.status_code != 200:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        return EvaluationResult(
            case_id=case.case_id,
            passed=False,
            reasons=[detail or f"API returned HTTP {response.status_code} from /chat."],
            tool_names=[],
            observation_count=0,
            answer_length=0,
            duration_seconds=time.monotonic() - started_at,
        )

    if not isinstance(payload, dict):
        return EvaluationResult(
            case_id=case.case_id,
            passed=False,
            reasons=["API did not return a JSON object."],
            tool_names=[],
            observation_count=0,
            answer_length=0,
            duration_seconds=time.monotonic() - started_at,
        )

    result = score_case(case, payload)
    result.duration_seconds = time.monotonic() - started_at
    return result


def print_summary(results: list[EvaluationResult], *, show_timing: bool) -> None:
    """Print a concise evaluation summary for operators."""
    passed_count = sum(1 for result in results if result.passed)
    total = len(results)

    print("Weekend Wizard evaluation summary")
    print(f"- passed: {passed_count}/{total}")
    print(f"- failed: {total - passed_count}/{total}")
    if show_timing and results:
        total_duration = sum(result.duration_seconds for result in results)
        slowest = max(results, key=lambda result: result.duration_seconds)
        average_duration = total_duration / len(results)
        print(f"- total_duration: {total_duration:.1f}s")
        print(f"- average_case_duration: {average_duration:.1f}s")
        print(f"- slowest_case: {slowest.case_id} ({slowest.duration_seconds:.1f}s)")

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        line = (
            f"[{status}] {result.case_id}: observations={result.observation_count} "
            f"tools={', '.join(result.tool_names) if result.tool_names else 'none'} "
            f"answer_length={result.answer_length}"
        )
        if show_timing:
            line += f" duration={result.duration_seconds:.1f}s"
        print(line)
        for reason in result.reasons:
            print(f"  - {reason}")


def main() -> None:
    """Run the Weekend Wizard evaluation workflow."""
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    base_url = args.api_url.rstrip("/")
    headers = get_api_headers()
    cases = load_cases(Path(args.cases_path))
    process: subprocess.Popen[str] | None = None
    results: list[EvaluationResult] = []

    try:
        if api_is_reachable(base_url, headers):
            print(f"Using existing Weekend Wizard API at {base_url}")
        else:
            if args.no_start_api:
                raise RuntimeError(
                    f"Weekend Wizard API is not reachable at {base_url} and --no-start-api was set."
                )
            print(f"Starting local Weekend Wizard API at {base_url}")
            process = start_local_api(project_dir)

        readiness = wait_for_ready(
            base_url=base_url,
            headers=headers,
            timeout_seconds=args.startup_timeout,
            poll_interval=args.poll_interval,
            process=process,
        )
        print(
            "API ready:"
            f" model={readiness.get('model_name')} tool_count={readiness.get('tool_count')}"
        )

        for case in cases:
            print(f"Evaluating: {case.case_id} ({case.category})")
            result = evaluate_case(base_url, headers, case, args.request_timeout)
            results.append(result)
            if args.fail_fast and not result.passed:
                break

        print_summary(results, show_timing=args.timing)
        if not all(result.passed for result in results):
            raise SystemExit(1)
    finally:
        if process is not None:
            stop_process(process)


if __name__ == "__main__":
    main()
