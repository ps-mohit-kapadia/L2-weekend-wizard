from __future__ import annotations

"""Typed application settings loaded from environment variables."""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Tuple


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with a fallback value.

    Args:
        name: The environment variable name to inspect.
        default: The value to use when the variable is unset.

    Returns:
        The parsed boolean value.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback value.

    Args:
        name: The environment variable name to inspect.
        default: The value to use when the variable is unset.

    Returns:
        The parsed integer value.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with a fallback value.

    Args:
        name: The environment variable name to inspect.
        default: The value to use when the variable is unset.

    Returns:
        The parsed floating-point value.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


@dataclass(frozen=True)
class HttpSettings:
    """HTTP behavior for external API calls.

    Attributes:
        request_timeout: Per-request timeout in seconds.
        max_retries: Number of retry attempts after the first failed request.
        retry_backoff_seconds: Base delay used for exponential retry backoff.
    """

    request_timeout: int = 20
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5


@dataclass(frozen=True)
class LlmSettings:
    """Settings for the local Ollama runtime.

    Attributes:
        ollama_url: HTTP endpoint used for Ollama chat requests.
        preferred_models: Preferred model names in descending priority order.
    """

    ollama_url: str = "http://127.0.0.1:11434/api/chat"
    preferred_models: Tuple[str, ...] = field(
        default_factory=lambda: ("mistral:7b", "llama3.2:latest", "llama3.2")
    )


@dataclass(frozen=True)
class AgentSettings:
    """Runtime behavior for the agent loop.

    Attributes:
        max_steps: Maximum number of decision iterations per interaction.
    """

    max_steps: int = 7


@dataclass(frozen=True)
class LoggingSettings:
    """Logging configuration for the application runtime.

    Attributes:
        level: Minimum log level emitted by the application logger hierarchy.
    """

    level: str = "WARNING"


@dataclass(frozen=True)
class AppSettings:
    """Top-level typed settings object for the application.

    Attributes:
        http: HTTP and retry configuration.
        llm: Local model runtime configuration.
        agent: Agent loop behavior configuration.
        logging: Logging configuration shared across the application.
    """

    http: HttpSettings
    llm: LlmSettings
    agent: AgentSettings
    logging: LoggingSettings


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load and cache application settings from environment variables.

    Returns:
        The fully populated application settings object.
    """
    return AppSettings(
        http=HttpSettings(
            request_timeout=_env_int("WEEKEND_WIZARD_REQUEST_TIMEOUT", 20),
            max_retries=_env_int("WEEKEND_WIZARD_HTTP_MAX_RETRIES", 2),
            retry_backoff_seconds=_env_float("WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS", 0.5),
        ),
        llm=LlmSettings(
            ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat"),
        ),
        agent=AgentSettings(
            max_steps=_env_int("WEEKEND_WIZARD_MAX_STEPS", 7),
        ),
        logging=LoggingSettings(
            level=os.getenv("WEEKEND_WIZARD_LOG_LEVEL", "WARNING").upper(),
        ),
    )
