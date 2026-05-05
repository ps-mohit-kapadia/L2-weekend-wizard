from __future__ import annotations

"""Application settings loaded from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple
from urllib.parse import urlsplit, urlunsplit


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


def _derive_ollama_tags_url(chat_url: str) -> str:
    """Derive a reasonable Ollama tags endpoint from the configured chat URL."""
    stripped = chat_url.rstrip("/")
    parsed = urlsplit(stripped)
    path = parsed.path.rstrip("/")

    if path.endswith("/api/chat"):
        tags_path = f"{path[:-len('/api/chat')]}/api/tags"
    elif path.endswith("/chat"):
        tags_path = f"{path[:-len('/chat')]}/tags"
    else:
        tags_path = f"{path}/api/tags" if path else "/api/tags"

    return urlunsplit((parsed.scheme, parsed.netloc, tags_path, parsed.query, parsed.fragment))


@dataclass(frozen=True)
class Settings:
    """Typed application settings for the Weekend Wizard runtime."""

    request_timeout: int
    tool_http_timeout: int
    http_max_retries: int
    http_retry_backoff_seconds: float
    ollama_url: str
    ollama_tags_url: str
    api_key: str
    api_host: str
    api_port: int
    api_url: str
    preferred_models: Tuple[str, ...]
    observability_mode: str
    log_level: str


def _env_tuple(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    """Read a comma-separated environment variable into a normalized tuple."""
    value = os.getenv(name)
    if value is None:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or default


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load and cache application settings from environment variables."""
    _load_dotenv()
    api_host = os.getenv("WEEKEND_WIZARD_API_HOST", "127.0.0.1").strip() or "127.0.0.1"
    api_port = _env_int("WEEKEND_WIZARD_API_PORT", 8000)
    ollama_chat_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat").rstrip("/")
    return Settings(
        request_timeout=_env_int("WEEKEND_WIZARD_REQUEST_TIMEOUT", 20),
        tool_http_timeout=_env_int("WEEKEND_WIZARD_TOOL_HTTP_TIMEOUT", 15),
        http_max_retries=_env_int("WEEKEND_WIZARD_HTTP_MAX_RETRIES", 2),
        http_retry_backoff_seconds=_env_float(
            "WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS",
            0.5,
        ),
        ollama_url=ollama_chat_url,
        ollama_tags_url=os.getenv("OLLAMA_TAGS_URL", _derive_ollama_tags_url(ollama_chat_url)).rstrip("/"),
        api_key=os.getenv("WEEKEND_WIZARD_API_KEY", "").strip(),
        api_host=api_host,
        api_port=api_port,
        api_url=os.getenv("WEEKEND_WIZARD_API_URL", f"http://{api_host}:{api_port}").rstrip("/"),
        preferred_models=_env_tuple("WEEKEND_WIZARD_PREFERRED_MODELS", ("llama3.1:8b",)),
        observability_mode=os.getenv("WEEKEND_WIZARD_OBSERVABILITY_MODE", "local").strip().lower(),
        log_level=os.getenv("WEEKEND_WIZARD_LOG_LEVEL", "WARNING").upper(),
    )
