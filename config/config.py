from __future__ import annotations

"""Application settings loaded from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple


def _load_dotenv() -> None:
    """Load simple key/value pairs from a local .env file when present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip("\"'")


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
    api_key: str
    preferred_models: Tuple[str, ...]
    observability_mode: str
    log_level: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings from environment variables."""
    _load_dotenv()
    return Settings(
        request_timeout=_env_int("WEEKEND_WIZARD_REQUEST_TIMEOUT", 20),
        http_max_retries=_env_int("WEEKEND_WIZARD_HTTP_MAX_RETRIES", 2),
        http_retry_backoff_seconds=_env_float(
            "WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS",
            0.5,
        ),
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"),
        api_key=os.getenv("WEEKEND_WIZARD_API_KEY", "").strip(),
        preferred_models=("llama3.1:8b",),
        # preferred_models=("qwen3.5:4b",),
        observability_mode=os.getenv("WEEKEND_WIZARD_OBSERVABILITY_MODE", "local").strip().lower(),
        log_level=os.getenv("WEEKEND_WIZARD_LOG_LEVEL", "WARNING").upper(),
    )
