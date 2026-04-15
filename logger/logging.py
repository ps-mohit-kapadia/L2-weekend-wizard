from __future__ import annotations

"""Structured logging helpers for the Weekend Wizard application."""

import json
import logging
from typing import Any, Mapping, Optional

from config.config import get_settings
from logger.context import get_log_context


_CONFIGURED = False


def configure_logging(level_name: Optional[str] = None) -> None:
    """Initialize process-wide logging once for the application.

    Args:
        level_name: Optional explicit log level override.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings = get_settings()
    resolved_name = (level_name or settings.logging.level or "WARNING").upper()
    level = getattr(logging, resolved_name, logging.WARNING)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _CONFIGURED = True


def _serialize_log_value(value: Any) -> str:
    """Serialize one structured log field value into a compact string.

    Args:
        value: The field value to serialize.

    Returns:
        A compact string representation suitable for structured log messages.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value)
    return json.dumps(value, ensure_ascii=True, default=str)


def format_event(event: str, **fields: Any) -> str:
    """Build a consistent event-style log message.

    Args:
        event: Event name describing the log record.
        **fields: Structured key-value details to append to the event.

    Returns:
        A single-line structured log message.
    """
    parts = [f"event={event}"]
    for key, value in fields.items():
        parts.append(f"{key}={_serialize_log_value(value)}")
    return " ".join(parts)


class ProjectLogger:
    """Wrap a standard logger with a structured application logging API.

    Args:
        logger: The underlying standard-library logger to emit through.
        layer: Logical application layer for the logger, such as ``api`` or
            ``orchestrator``.
        bound_fields: Optional fields that should be included on every record.
    """

    def __init__(
        self,
        logger: logging.Logger,
        *,
        layer: str | None = None,
        bound_fields: Mapping[str, Any] | None = None,
    ) -> None:
        self._logger = logger
        self._layer = layer
        self._bound_fields = dict(bound_fields or {})

    @property
    def name(self) -> str:
        """Return the fully qualified logger name.

        Returns:
            The underlying standard-library logger name.
        """
        return self._logger.name

    def bind(self, **fields: Any) -> ProjectLogger:
        """Create a derived logger with additional bound fields.

        Args:
            **fields: Extra structured fields to include on every record.

        Returns:
            A new project logger sharing the same destination logger.
        """
        merged = dict(self._bound_fields)
        merged.update(fields)
        return ProjectLogger(self._logger, layer=self._layer, bound_fields=merged)

    def debug(self, event: str, **fields: Any) -> None:
        """Emit a debug-level structured log record.

        Args:
            event: Event name describing the record.
            **fields: Structured fields attached to the record.
        """
        self._emit(logging.DEBUG, event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        """Emit an info-level structured log record.

        Args:
            event: Event name describing the record.
            **fields: Structured fields attached to the record.
        """
        self._emit(logging.INFO, event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        """Emit a warning-level structured log record.

        Args:
            event: Event name describing the record.
            **fields: Structured fields attached to the record.
        """
        self._emit(logging.WARNING, event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        """Emit an error-level structured log record.

        Args:
            event: Event name describing the record.
            **fields: Structured fields attached to the record.
        """
        self._emit(logging.ERROR, event, **fields)

    def exception(self, event: str, **fields: Any) -> None:
        """Emit an exception record with traceback information.

        Args:
            event: Event name describing the exception.
            **fields: Structured fields attached to the record.
        """
        self._logger.exception(format_event(event, **self._build_fields(fields)))

    def _emit(self, level: int, event: str, **fields: Any) -> None:
        """Emit one structured log record through the underlying logger.

        Args:
            level: Standard logging level constant.
            event: Event name describing the record.
            **fields: Structured fields attached to the record.
        """
        self._logger.log(level, format_event(event, **self._build_fields(fields)))

    def _build_fields(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        """Build the final structured field map for one record.

        Args:
            fields: Per-record fields provided by the caller.

        Returns:
            The merged structured field dictionary in log output order.
        """
        structured_fields: dict[str, Any] = {}
        if self._layer is not None:
            structured_fields["layer"] = self._layer

        for key, value in get_log_context().items():
            if value is not None:
                structured_fields[key] = value

        for key, value in self._bound_fields.items():
            if value is not None:
                structured_fields[key] = value

        for key, value in fields.items():
            if value is not None:
                structured_fields[key] = value

        return structured_fields


def get_logger(
    name: str,
    *,
    layer: str | None = None,
    **bound_fields: Any,
) -> ProjectLogger:
    """Return a structured application logger under the project namespace.

    Args:
        name: Logger name or suffix within the project namespace.
        layer: Optional logical application layer name for structured logs.
        **bound_fields: Optional fields that should be attached to every record.

    Returns:
        A structured project logger configured for the requested namespace.
    """
    configure_logging()
    if name.startswith("weekend_wizard."):
        logger = logging.getLogger(name)
    else:
        logger = logging.getLogger(f"weekend_wizard.{name}")
    return ProjectLogger(logger, layer=layer, bound_fields=bound_fields)
