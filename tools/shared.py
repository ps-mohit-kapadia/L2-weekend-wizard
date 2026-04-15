from __future__ import annotations

"""Shared HTTP helpers used by MCP tool implementations."""

import time
from typing import Any, Dict

import requests

from config.config import get_settings
from logger.logging import get_logger


logger = get_logger("tools.shared", layer="tools")


def get_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Fetch a JSON payload from an HTTP endpoint with retry handling.

    Args:
        url: The target endpoint URL.
        params: Optional query parameters for the request.

    Returns:
        The parsed JSON response body.

    Raises:
        requests.RequestException: If all retry attempts fail.
    """
    settings = get_settings()
    logger.info("request_start", url=url, params=params)

    last_exception: Exception | None = None
    for attempt in range(settings.http.max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=settings.http.request_timeout)
            response.raise_for_status()
            logger.info("request_success", url=url, status=response.status_code)
            return response.json()
        except requests.RequestException as exc:
            last_exception = exc
            if attempt >= settings.http.max_retries:
                break

            delay = settings.http.retry_backoff_seconds * (2 ** attempt)
            logger.warning(
                "request_retry",
                url=url,
                attempt=attempt + 1,
                delay=delay,
                details=str(exc),
            )
            time.sleep(delay)

    assert last_exception is not None
    raise last_exception


def error_payload(source: str, exc: Exception) -> Dict[str, str]:
    """Build a consistent error payload for failed tool requests.

    Args:
        source: Logical source name for the failed request.
        exc: The exception raised during request processing.

    Returns:
        A serializable error payload for tool responses.
    """
    logger.warning("request_failure", source=source, details=str(exc))
    return {"error": f"{source} request failed", "details": str(exc)}
