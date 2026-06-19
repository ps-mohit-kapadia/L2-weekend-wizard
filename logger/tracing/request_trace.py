from __future__ import annotations

"""Minimal request-scoped execution tracing for local debugging."""

from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
from secrets import token_hex
from typing import Any, Dict, List


@dataclass
class TraceEvent:
    """One append-only execution trace event."""

    timestamp: datetime
    event: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RequestTrace:
    """Collect chronological debugging events for one request."""

    request_id: str
    user_prompt: str
    started_at: datetime
    events: List[TraceEvent] = field(default_factory=list)

    def add_event(self, event: str, **data: Any) -> None:
        """Append one new trace event without mutating prior entries."""
        self.events.append(
            TraceEvent(
                timestamp=datetime.now(),
                event=event,
                data=dict(data),
            )
        )


def create_trace(prompt: str) -> RequestTrace:
    """Create a request trace with a lightweight request id."""
    trace = RequestTrace(
        request_id=f"req_{token_hex(4)}",
        user_prompt=prompt,
        started_at=datetime.now(),
    )
    trace.add_event("interaction_started")
    return trace


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
        f"REQUEST TRACE: {trace.request_id}",
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
            trace.request_id,
            target,
            exc,
        )
