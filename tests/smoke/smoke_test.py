from __future__ import annotations

"""End-to-end smoke test for the Weekend Wizard API."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from config.config import get_settings
DEFAULT_PROMPT = "Tell me a joke."
API_KEY_HEADER = "X-API-Key"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the smoke test."""
    parser = argparse.ArgumentParser(
        description="Run an end-to-end smoke test against the Weekend Wizard API."
    )
    parser.add_argument(
        "--api-url",
        default=get_settings().api_url,
        help="Base URL of the FastAPI service. Default: configured WEEKEND_WIZARD_API_URL/runtime host+port.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help=f"Prompt sent to /chat. Default: {DEFAULT_PROMPT!r}",
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
        "--no-start-api",
        action="store_true",
        help="Fail instead of starting a local API process when the API is not already reachable.",
    )
    return parser.parse_args()


def get_api_headers() -> dict[str, str]:
    """Return required headers for protected API requests."""
    api_key = os.getenv("WEEKEND_WIZARD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("WEEKEND_WIZARD_API_KEY is not configured for the smoke test.")
    return {API_KEY_HEADER: api_key}


def api_is_reachable(base_url: str, timeout: int = 3) -> bool:
    """Return whether the API health endpoint is reachable."""
    try:
        response = requests.get(f"{base_url}/health", timeout=timeout)
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


def validate_chat_payload_shape(payload: dict[str, Any]) -> None:
    """Validate the structural shape of the /chat response payload."""
    answer = payload.get("answer")
    tool_observations = payload.get("tool_observations")

    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("Smoke test failed: /chat response did not include a non-empty answer.")
    if not isinstance(tool_observations, list):
        raise RuntimeError("Smoke test failed: /chat response did not include a tool_observations list.")


def validate_chat_response_status(payload: dict[str, Any]) -> None:
    """Validate the operator-facing quality status of the /chat response payload."""
    response_status = payload.get("response_status")

    if response_status == "degraded":
        raise RuntimeError("Smoke test failed: /chat response was marked degraded by the backend.")


def start_local_api(project_dir: Path) -> subprocess.Popen[str]:
    """Start a local Weekend Wizard API process."""
    command = [sys.executable, "main.py", "api"]
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


def main() -> None:
    """Run the end-to-end smoke test."""
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[2]
    base_url = args.api_url.rstrip("/")
    headers = get_api_headers()
    process: subprocess.Popen[str] | None = None

    try:
        if api_is_reachable(base_url):
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

        response = requests.post(
            f"{base_url}/chat",
            json={"prompt": args.prompt},
            headers=headers,
            timeout=120,
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Weekend Wizard API returned invalid JSON from /chat.") from exc

        if response.status_code != 200:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise RuntimeError(detail or f"Weekend Wizard API returned HTTP {response.status_code}.")

        if not isinstance(payload, dict):
            raise RuntimeError("Smoke test failed: /chat did not return a JSON object.")

        validate_chat_payload_shape(payload)
        validate_chat_response_status(payload)

        print("Smoke test passed.")
        print(f"Prompt: {args.prompt}")
        print(f"Answer: {payload['answer'][:300]}")
        print(f"Tool observations: {len(payload['tool_observations'])}")
    finally:
        if process is not None:
            stop_process(process)


if __name__ == "__main__":
    main()
