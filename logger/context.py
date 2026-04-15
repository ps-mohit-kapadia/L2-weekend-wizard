from __future__ import annotations

"""Context propagation helpers for structured application logging."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator
from uuid import uuid4


_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "weekend_wizard_log_context",
    default={},
)


def get_log_context() -> dict[str, Any]:
    """Return the currently bound structured log context.

    Returns:
        A shallow copy of the currently active context fields.
    """
    return dict(_LOG_CONTEXT.get())


def make_request_id(prefix: str) -> str:
    """Create a compact request identifier for one interaction flow.

    Args:
        prefix: Short source prefix such as ``streamlit`` or ``api``.

    Returns:
        A compact request identifier.
    """
    return f"{prefix}-{uuid4().hex[:8]}"


@contextmanager
def bind_log_context(**fields: Any) -> Iterator[dict[str, Any]]:
    """Temporarily bind structured fields to the active logging context.

    Args:
        **fields: Structured fields that should be attached to nested logs.

    Yields:
        The merged structured logging context visible inside the block.
    """
    merged = get_log_context()
    merged.update({key: value for key, value in fields.items() if value is not None})
    token = _LOG_CONTEXT.set(merged)
    try:
        yield merged
    finally:
        _LOG_CONTEXT.reset(token)


@contextmanager
def request_context(prefix: str, **fields: Any) -> Iterator[str]:
    """Bind a fresh request identifier to the active logging context.

    Args:
        prefix: Short source prefix such as ``streamlit`` or ``api``.
        **fields: Additional structured fields to bind for the request scope.

    Yields:
        The generated request identifier.
    """
    request_id = fields.pop("request_id", None) or make_request_id(prefix)
    with bind_log_context(request_id=request_id, **fields):
        yield request_id


@contextmanager
def ensure_request_context(prefix: str, **fields: Any) -> Iterator[str]:
    """Ensure that a request identifier exists for the current execution scope.

    Args:
        prefix: Short source prefix used when a new request id must be created.
        **fields: Additional structured fields to bind for the active scope.

    Yields:
        The existing or newly created request identifier.
    """
    current_request_id = get_log_context().get("request_id")
    if current_request_id is not None:
        with bind_log_context(**fields):
            yield str(current_request_id)
        return

    with request_context(prefix, **fields) as request_id:
        yield request_id
