from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from logger.tracing.request_trace import create_trace, render_trace, trace_llm_decision, truncate_repr, write_trace


class RequestTraceTests(unittest.TestCase):
    def test_create_trace_adds_interaction_started_event(self) -> None:
        trace = create_trace("hello world")

        self.assertTrue(trace.correlation_id.startswith("corr_"))
        self.assertEqual(trace.user_prompt, "hello world")
        self.assertEqual(len(trace.events), 1)
        self.assertEqual(trace.events[0].event, "interaction_started")

    def test_render_trace_includes_timestamps_and_duration(self) -> None:
        trace = create_trace("hello world")
        trace.add_event("llm_call_completed", model="demo", duration_ms=123, messages_count=2)

        rendered = render_trace(trace)

        self.assertIn("CORRELATION ID:", rendered)
        self.assertIn("STARTED:", rendered)
        self.assertIn("ENDED:", rendered)
        self.assertIn("TOTAL DURATION:", rendered)
        self.assertIn("EVENT: interaction_started", rendered)
        self.assertIn("EVENT: llm_call_completed", rendered)
        self.assertIn("* duration_ms: 123", rendered)

    def test_trace_llm_decision_adds_canonical_event(self) -> None:
        trace = create_trace("hello world")

        trace_llm_decision(
            trace,
            phase="react",
            step_number=1,
            action="tool",
            tool_name="get_weather",
            accepted=True,
            reason="tool_selected",
        )

        event = trace.events[-1]
        self.assertEqual(event.event, "llm_decision")
        self.assertEqual(event.data["phase"], "react")
        self.assertEqual(event.data["step_number"], 1)
        self.assertEqual(event.data["tool_name"], "get_weather")
        self.assertTrue(event.data["accepted"])
        self.assertEqual(event.data["reason"], "tool_selected")

    def test_truncate_repr_limits_long_values(self) -> None:
        text = "x" * 1000

        rendered = truncate_repr(text, limit=20)

        self.assertLessEqual(len(rendered), 20)
        self.assertTrue(rendered.endswith("..."))

    def test_write_trace_appends_rendered_trace(self) -> None:
        trace = create_trace("hello world")

        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "trace.log"
            write_trace(trace, log_path)

            contents = log_path.read_text(encoding="utf-8")

        self.assertIn(f"CORRELATION ID: {trace.correlation_id}", contents)
        self.assertTrue(contents.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
