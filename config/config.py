from __future__ import annotations

"""Application settings loaded from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _env_positive_int(name: str, default: int) -> int:
    """Read a positive integer environment variable with a fallback value."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid integer greater than 0.") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return parsed


def _env_non_negative_int(name: str, default: int) -> int:
    """Read a non-negative integer environment variable with a fallback value."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid integer greater than or equal to 0.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")
    return parsed


def _env_non_negative_float(name: str, default: float) -> float:
    """Read a non-negative float environment variable with a fallback value."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid number greater than or equal to 0.") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be greater than or equal to 0.")
    return parsed


def _env_log_level(name: str, default: str) -> str:
    """Read and validate a log level environment variable with a fallback value."""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.upper()
    if normalized not in _VALID_LOG_LEVELS:
        allowed = ", ".join(sorted(_VALID_LOG_LEVELS))
        raise ValueError(f"{name} must be one of {allowed}.")
    return normalized


@dataclass(frozen=True)
class Settings:
    """Typed application settings for the Weekend Wizard runtime."""

    request_timeout: int
    tool_http_timeout: int
    http_max_retries: int
    http_retry_backoff_seconds: float
    ollama_url: str
    preferred_models: Tuple[str, ...]
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings from environment variables."""
    return Settings(
        request_timeout=_env_positive_int("WEEKEND_WIZARD_REQUEST_TIMEOUT", 1200),
        tool_http_timeout=_env_positive_int("WEEKEND_WIZARD_TOOL_HTTP_TIMEOUT", 20),
        http_max_retries=_env_non_negative_int("WEEKEND_WIZARD_HTTP_MAX_RETRIES", 2),
        http_retry_backoff_seconds=_env_non_negative_float(
            "WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS",
            0.5,
        ),
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"),
        preferred_models=("llama3.1:8b",),
        log_level=_env_log_level("WEEKEND_WIZARD_LOG_LEVEL", "WARNING"),
    )
