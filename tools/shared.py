from __future__ import annotations

"""Shared HTTP helpers used by MCP tool implementations."""

import asyncio
from typing import Any, Dict

import requests

from config.config import get_settings
from logger.logging import get_logger


logger = get_logger("tools.shared")


async def get_json(url: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
    logger.info("Starting HTTP request to %s with params=%s", url, params)

    last_exception: Exception | None = None
    for attempt in range(settings.http_max_retries + 1):
        try:
            response = await asyncio.to_thread(
                requests.get,
                url,
                params=params,
                timeout=settings.tool_http_timeout,
            )
            response.raise_for_status()
            logger.info("HTTP request succeeded for %s with status %s", url, response.status_code)
            return response.json()
        except requests.RequestException as exc:
            last_exception = exc
            if attempt >= settings.http_max_retries:
                break

            delay = settings.http_retry_backoff_seconds * (2 ** attempt)
            logger.warning(
                "Retrying HTTP request to %s (attempt %d, delay %.1fs): %s",
                url,
                attempt + 1,
                delay,
                exc,
            )
            await asyncio.sleep(delay)

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
    logger.warning("%s request failed: %s", source, exc)
    return {"error": f"{source} request failed", "details": str(exc)}
