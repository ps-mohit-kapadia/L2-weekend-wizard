from __future__ import annotations

"""Reusable logger configuration for Weekend Wizard observability modes."""

import contextvars
import logging
import sys
from typing import Any

from config.config import get_settings


_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_KNOWN_EXTRA_FIELDS = (
    "event",
    "phase",
    "outcome",
    "duration_ms",
    "tool_name",
    "model_name",
    "status_code",
)


def set_request_context(request_id: str) -> contextvars.Token[str | None]:
    """Bind a request ID to the current execution context."""
    return _REQUEST_ID.set(request_id)


def reset_request_context(token: contextvars.Token[str | None]) -> None:
    """Reset the current request ID context."""
    _REQUEST_ID.reset(token)


def observability_mode() -> str:
    """Return the configured observability mode."""
    return get_settings().observability_mode


def telemetry_enabled() -> bool:
    """Return whether richer observability telemetry should be emitted."""
    return observability_mode() in {"staging", "production"}


def staging_mode() -> bool:
    """Return whether the app is running with staging observability."""
    return observability_mode() == "staging"


def production_mode() -> bool:
    """Return whether the app is running with production observability."""
    return observability_mode() == "production"


class _RequestContextFilter(logging.Filter):
    """Attach request-scoped observability context to each record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _REQUEST_ID.get()
        record.observability_mode = observability_mode()
        for field in _KNOWN_EXTRA_FIELDS:
            if not hasattr(record, field):
                setattr(record, field, None)
        return True


class _ObservabilityFormatter(logging.Formatter):
    """Mode-aware formatter that preserves readable logs and enriches telemetry."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        mode = getattr(record, "observability_mode", "local")
        if mode == "local":
            return message

        extras: list[str] = []
        request_id = getattr(record, "request_id", None)
        if request_id:
            extras.append(f"request_id={request_id}")

        for field in _KNOWN_EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is None:
                continue
            extras.append(f"{field}={value}")

        if not extras:
            return message
        return f"{message} | " + " ".join(extras)


def get_log_extra(**kwargs: Any) -> dict[str, Any]:
    """Return structured extra fields for observability-aware log calls."""
    return {key: value for key, value in kwargs.items() if value is not None}


def get_logger(name: str, *_, **__) -> logging.Logger:
    """Create and return a configured logger instance."""
    if not name.startswith("weekend_wizard."):
        name = f"weekend_wizard.{name}"

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    level_name = get_settings().log_level
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    # MCP stdio uses stdout for JSON-RPC, so logs must stay on stderr.
    handler = logging.StreamHandler(sys.stderr)
    formatter = _ObservabilityFormatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.addFilter(_RequestContextFilter())

    logger.addHandler(handler)
    logger.propagate = False

    return logger
