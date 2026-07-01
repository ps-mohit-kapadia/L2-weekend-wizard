from __future__ import annotations

"""Application settings loaded from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple

from dotenv import load_dotenv

load_dotenv(override=False)

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_VALID_LLM_PROVIDERS = {"ollama", "aiplatform"}


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


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a fallback value."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid number.") from exc


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


def _env_llm_provider(name: str, default: str) -> str:
    """Read and validate the configured LLM provider."""
    value = os.getenv(name, default).lower()
    if value not in _VALID_LLM_PROVIDERS:
        allowed = ", ".join(sorted(_VALID_LLM_PROVIDERS))
        raise ValueError(f"{name} must be one of {allowed}.")
    return value


@dataclass(frozen=True)
class Settings:
    """Typed application settings for the Weekend Wizard runtime."""

    request_timeout: int
    tool_http_timeout: int
    http_max_retries: int
    http_retry_backoff_seconds: float
    llm_provider: str
    ollama_url: str
    ollama_tags_url: str
    aiplatform_api_key: str | None
    aiplatform_base_url: str
    aiplatform_timeout: int
    aiplatform_chat_path: str
    preferred_models: Tuple[str, ...]
    log_level: str
    api_key: str | None
    max_prompt_chars: int
    rate_limit_requests: int
    rate_limit_window_seconds: int
    api_host: str
    api_port: int
    api_url: str
    max_react_steps: int
    claude_sdk_max_turns: int
    react_temperature: float
    repair_temperature: float
    startup_retry_interval_seconds: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings from environment variables."""
    api_host = os.getenv("WEEKEND_WIZARD_API_HOST", "127.0.0.1")
    api_port = _env_positive_int("WEEKEND_WIZARD_API_PORT", 8000)
    api_url = os.getenv("WEEKEND_WIZARD_API_URL", f"http://{api_host}:{api_port}")

    return Settings(
        request_timeout=_env_positive_int("WEEKEND_WIZARD_REQUEST_TIMEOUT", 1200),
        tool_http_timeout=_env_positive_int("WEEKEND_WIZARD_TOOL_HTTP_TIMEOUT", 20),
        http_max_retries=_env_non_negative_int("WEEKEND_WIZARD_HTTP_MAX_RETRIES", 2),
        http_retry_backoff_seconds=_env_non_negative_float(
            "WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS",
            0.5,
        ),
        llm_provider=_env_llm_provider("LLM_PROVIDER", "ollama"),
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"),
        ollama_tags_url=os.getenv("OLLAMA_TAGS_URL", "http://127.0.0.1:11434/api/tags"),
        aiplatform_api_key=os.getenv("AIPLATFORM_API_KEY"),
        aiplatform_base_url=os.getenv(
            "AIPLATFORM_BASE_URL", "https://aiapidev.3ecompany.com"
        ).rstrip("/"),
        aiplatform_timeout=_env_positive_int("AIPLATFORM_TIMEOUT", 120),
        aiplatform_chat_path=os.getenv(
            "AIPLATFORM_CHAT_PATH", "/v1/chat/completions"
        ),
        preferred_models=(os.getenv("MODEL", "llama3.1:8b"),),
        log_level=_env_log_level("WEEKEND_WIZARD_LOG_LEVEL", "WARNING"),
        api_key=os.getenv("WEEKEND_WIZARD_API_KEY"),
        max_prompt_chars=_env_positive_int("WEEKEND_WIZARD_MAX_PROMPT_CHARS", 4000),
        rate_limit_requests=_env_positive_int("WEEKEND_WIZARD_RATE_LIMIT_REQUESTS", 20),
        rate_limit_window_seconds=_env_positive_int("WEEKEND_WIZARD_RATE_LIMIT_WINDOW_SECONDS", 60),
        api_host=api_host,
        api_port=api_port,
        api_url=api_url,
        max_react_steps=_env_positive_int("WEEKEND_WIZARD_MAX_REACT_STEPS", 6),
        claude_sdk_max_turns=_env_positive_int("CLAUDE_SDK_MAX_TURNS", 4),
        react_temperature=_env_float("WEEKEND_WIZARD_REACT_TEMPERATURE", 0.2),
        repair_temperature=_env_float("WEEKEND_WIZARD_REPAIR_TEMPERATURE", 0.0),
        startup_retry_interval_seconds=_env_float("WEEKEND_WIZARD_STARTUP_RETRY_INTERVAL_SECONDS", 5.0),
    )
