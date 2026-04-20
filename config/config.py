from __future__ import annotations

"""Application settings loaded from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback value."""
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a fallback value."""
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    """Typed application settings for the Weekend Wizard runtime."""

    request_timeout: int
    http_max_retries: int
    http_retry_backoff_seconds: float
    ollama_url: str
    preferred_models: Tuple[str, ...]
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings from environment variables."""
    return Settings(
        request_timeout=_env_int("WEEKEND_WIZARD_REQUEST_TIMEOUT", 20),
        http_max_retries=_env_int("WEEKEND_WIZARD_HTTP_MAX_RETRIES", 2),
        http_retry_backoff_seconds=_env_float(
            "WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS",
            0.5,
        ),
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"),
        preferred_models=("llama3.1:8b",),
        # preferred_models=("qwen3.5:4b",),
        log_level=os.getenv("WEEKEND_WIZARD_LOG_LEVEL", "WARNING").upper(),
    )
