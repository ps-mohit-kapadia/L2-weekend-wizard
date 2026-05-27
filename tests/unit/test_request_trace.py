from __future__ import annotations

import unittest

from logger.tracing.request_trace import create_trace, render_trace, truncate_repr


class RequestTraceTests(unittest.TestCase):
    def test_create_trace_adds_interaction_started_event(self) -> None:
        trace = create_trace("hello world")

        self.assertTrue(trace.request_id.startswith("req_"))
        self.assertEqual(trace.user_prompt, "hello world")
        self.assertEqual(len(trace.events), 1)
        self.assertEqual(trace.events[0].event, "interaction_started")

    def test_render_trace_includes_timestamps_and_duration(self) -> None:
        trace = create_trace("hello world")
        trace.add_event("llm_call_completed", model="demo", duration_ms=123, messages_count=2)

        rendered = render_trace(trace)

        self.assertIn("REQUEST TRACE:", rendered)
        self.assertIn("STARTED:", rendered)
        self.assertIn("ENDED:", rendered)
        self.assertIn("TOTAL DURATION:", rendered)
        self.assertIn("EVENT: interaction_started", rendered)
        self.assertIn("EVENT: llm_call_completed", rendered)
        self.assertIn("* duration_ms: 123", rendered)

    def test_truncate_repr_limits_long_values(self) -> None:
        text = "x" * 1000

        rendered = truncate_repr(text, limit=20)

        self.assertLessEqual(len(rendered), 20)
        self.assertTrue(rendered.endswith("..."))


if __name__ == "__main__":
    unittest.main()
