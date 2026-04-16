from __future__ import annotations

import unittest

from schemas.agent import (
    ExecutionPlan,
    OrchestratorContext,
    PlanLocation,
    PlanStep,
    ReflectionResult,
    validate_execution_plan,
    validate_reflection_result,
)
from schemas.tools import BookResults, DogResult, GeoResult, ToolError, WeatherResult, parse_tool_payload


class SchemaTests(unittest.TestCase):
    def test_validate_execution_plan_parses_plan(self) -> None:
        plan = validate_execution_plan(
            {
                "goal": "joke",
                "requested_tools": ["random_joke"],
                "execution_steps": [{"tool": "random_joke", "args": {}}],
            }
        )

        self.assertIsInstance(plan, ExecutionPlan)
        self.assertIsInstance(plan.execution_steps[0], PlanStep)
        self.assertEqual(plan.execution_steps[0].tool, "random_joke")

    def test_validate_reflection_result_parses_answer(self) -> None:
        result = validate_reflection_result({"answer": "tightened"})

        self.assertIsInstance(result, ReflectionResult)
        self.assertEqual(result.answer, "tightened")

    def test_orchestrator_context_keeps_runtime_state(self) -> None:
        context = OrchestratorContext(
            tool_names=["get_weather", "random_joke"],
            history=[{"role": "user", "content": "prompt"}],
            model_name="demo-model",
        )

        self.assertEqual(context.tool_names, ["get_weather", "random_joke"])
        self.assertEqual(context.history[0]["role"], "user")
        self.assertEqual(context.model_name, "demo-model")

    def test_plan_location_supports_city_and_coordinates(self) -> None:
        location = PlanLocation(city="New York", latitude=40.7128, longitude=-74.0060)

        self.assertEqual(location.city, "New York")
        self.assertEqual(location.latitude, 40.7128)

    def test_parse_tool_payload_returns_typed_weather(self) -> None:
        payload = parse_tool_payload(
            "get_weather",
            {
                "latitude": 12.9,
                "longitude": 77.5,
                "temperature": 23.0,
                "temperature_unit": "C",
                "weather_summary": "clear sky",
            },
        )

        self.assertIsInstance(payload, WeatherResult)
        self.assertEqual(payload.temperature, 23.0)

    def test_parse_tool_payload_returns_typed_books(self) -> None:
        payload = parse_tool_payload(
            "book_recs",
            {
                "topic": "mystery",
                "count": 1,
                "results": [{"title": "Book One", "author": "Author A"}],
            },
        )

        self.assertIsInstance(payload, BookResults)
        self.assertEqual(payload.results[0].title, "Book One")

    def test_parse_tool_payload_returns_typed_tool_error(self) -> None:
        payload = parse_tool_payload(
            "random_dog",
            {"error": "dog request failed", "details": "timeout"},
        )

        self.assertIsInstance(payload, ToolError)
        self.assertEqual(payload.details, "timeout")

    def test_parse_tool_payload_returns_typed_dog_result(self) -> None:
        payload = parse_tool_payload(
            "random_dog",
            {"status": "success", "image_url": "https://example.com/dog.jpg"},
        )

        self.assertIsInstance(payload, DogResult)
        self.assertIn("dog.jpg", payload.image_url)

    def test_parse_tool_payload_returns_typed_geo_result(self) -> None:
        payload = parse_tool_payload(
            "city_to_coords",
            {
                "city": "New York",
                "latitude": 40.7128,
                "longitude": -74.0060,
                "country": "United States",
            },
        )

        self.assertIsInstance(payload, GeoResult)
        self.assertEqual(payload.city, "New York")


if __name__ == "__main__":
    unittest.main()
