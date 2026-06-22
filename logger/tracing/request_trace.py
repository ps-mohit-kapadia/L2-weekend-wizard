from __future__ import annotations

"""Minimal request-scoped execution tracing for local debugging."""

from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
import inspect
import logging
from pathlib import Path
from secrets import token_hex
import time
from typing import Any, Callable, Dict, List


@dataclass
class TraceEvent:
    """One append-only execution trace event."""

    timestamp: datetime
    event: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestTrace:
    """Collect chronological debugging events for one request."""

    correlation_id: str
    user_prompt: str
    started_at: datetime
    events: List[TraceEvent] = field(default_factory=list)
    _span_stack: List[str] = field(default_factory=list)
    _span_counter: int = 0

    def add_event(self, event: str, **data: Any) -> None:
        """Append one new trace event without mutating prior entries."""
        if self._span_stack and "span_id" not in data:
            data["span_id"] = self._span_stack[-1]
        self.events.append(
            TraceEvent(
                timestamp=datetime.now(),
                event=event,
                data=dict(data),
            )
        )

    def span(self, name: str, **attributes: Any) -> "TraceSpan":
        """Create one timed child span for a meaningful operation."""
        return TraceSpan(self, name, attributes)

    def _next_span_id(self) -> str:
        self._span_counter += 1
        return f"spn_{self._span_counter}"


class TraceSpan:
    """Timed trace span with parent-child relationship inside one request."""

    def __init__(self, trace: RequestTrace, name: str, attributes: Dict[str, Any]) -> None:
        self._trace = trace
        self._name = name
        self._attributes = attributes
        self._span_id = trace._next_span_id()
        self._parent_span_id = trace._span_stack[-1] if trace._span_stack else None
        self._started = 0.0

    def __enter__(self) -> "TraceSpan":
        self._started = time.perf_counter()
        data = {
            "span_id": self._span_id,
            "name": self._name,
            **self._attributes,
        }
        if self._parent_span_id:
            data["parent_span_id"] = self._parent_span_id
        self._trace.add_event("span_started", **data)
        self._trace._span_stack.append(self._span_id)
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if self._trace._span_stack and self._trace._span_stack[-1] == self._span_id:
            self._trace._span_stack.pop()
        duration_ms = int((time.perf_counter() - self._started) * 1000)
        data: Dict[str, Any] = {
            "span_id": self._span_id,
            "name": self._name,
            "duration_ms": duration_ms,
            "status": "error" if exc else "ok",
        }
        if self._parent_span_id:
            data["parent_span_id"] = self._parent_span_id
        if exc:
            data["error_type"] = type(exc).__name__
        self._trace.add_event("span_completed", **data)
        return False


def traced_span(
    name: str,
    attributes: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a function that accepts a ``trace`` argument with one timed span."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(func)

        def span_attributes(args: tuple[Any, ...], kwargs: Dict[str, Any]) -> tuple[RequestTrace | None, Dict[str, Any]]:
            bound = signature.bind_partial(*args, **kwargs)
            trace = bound.arguments.get("trace")
            if not isinstance(trace, RequestTrace):
                return None, {}
            return trace, attributes(bound.arguments) if attributes else {}

        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                trace, span_data = span_attributes(args, kwargs)
                if trace is None:
                    return await func(*args, **kwargs)
                with trace.span(name, **span_data):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            trace, span_data = span_attributes(args, kwargs)
            if trace is None:
                return func(*args, **kwargs)
            with trace.span(name, **span_data):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def create_correlation_id() -> str:
    """Create one app-wide correlation id for a single execution."""
    return f"corr_{token_hex(8)}"


def create_trace(prompt: str) -> RequestTrace:
    """Create a request trace with a lightweight request id."""
    trace = RequestTrace(
        correlation_id=create_correlation_id(),
        user_prompt=prompt,
        started_at=datetime.now(),
    )
    trace.add_event("interaction_started")
    return trace


def trace_llm_decision(
    trace: RequestTrace | None,
    *,
    phase: str,
    action: str,
    accepted: bool,
    reason: str,
    step_number: int | None = None,
    tool_name: str | None = None,
) -> None:
    """Append one canonical LLM decision event when tracing is enabled."""
    if trace is None:
        return

    data: Dict[str, Any] = {
        "phase": phase,
        "action": action,
        "accepted": accepted,
        "reason": reason,
    }
    if step_number is not None:
        data["step_number"] = step_number
    if tool_name:
        data["tool_name"] = tool_name
    trace.add_event("llm_decision", **data)


def truncate_repr(value: Any, limit: int = 300) -> str:
    """Return a compact repr truncated for debugging readability."""
    rendered = repr(value)
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[: max(0, limit - 3)]}..."


def render_trace(trace: RequestTrace) -> str:
    """Render one human-readable execution trace."""
    end_time = trace.events[-1].timestamp if trace.events else trace.started_at
    duration_ms = max(0, int((end_time - trace.started_at).total_seconds() * 1000))
    started_text = trace.started_at.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    ended_text = end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    lines = [
        "=" * 48,
        f"CORRELATION ID: {trace.correlation_id}",
        "=" * 48,
        f"STARTED: {started_text}",
        f"ENDED: {ended_text}",
        f"TOTAL DURATION: {duration_ms}ms",
        "",
        "USER:",
        trace.user_prompt,
        "",
    ]

    for event in trace.events:
        lines.append(
            f"[{event.timestamp.strftime('%H:%M:%S.%f')[:-3]}] EVENT: {event.event}"
        )
        if event.data:
            for key, value in event.data.items():
                lines.append(f"* {key}: {truncate_repr(value)}")
        lines.append("")

    lines.extend(
        [
            f"TOTAL EVENTS: {len(trace.events)}",
            f"TOTAL DURATION: {duration_ms}ms",
            "=" * 48,
        ]
    )
    return "\n".join(lines)


def write_trace(trace: RequestTrace, log_path: Path | None = None) -> None:
    """Append one rendered request trace to the local trace log."""
    target = log_path or Path("logs") / "trace.log"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as file:
            file.write(render_trace(trace))
            file.write("\n\n")
    except OSError as exc:
        logging.getLogger("weekend_wizard.logger.tracing.request_trace").warning(
            "Could not write request trace %s to %s: %s",
            trace.correlation_id,
            target,
            exc,
        )
