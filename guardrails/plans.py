from __future__ import annotations

"""Planner output validation helpers for Weekend Wizard."""

from typing import List

from guardrails.guardrails import parse_coords, requested_tools as infer_requested_tools
from schemas.agent import ExecutionPlan


def validate_plan_semantics(plan: ExecutionPlan, available_tools: List[str], user_prompt: str) -> None:
    """Validate planner output against supported runtime constraints.

    Args:
        plan: Typed execution plan returned by the planner.
        available_tools: Tool names exposed by the MCP runtime.
        user_prompt: Original user prompt used to infer supported intent.

    Raises:
        ValueError: If the plan requests unsupported tools, omits requested tools,
            or violates runtime dependency rules.
    """
    available = set(available_tools)
    user_requested = infer_requested_tools(user_prompt)
    coords_in_prompt = parse_coords(user_prompt)
    if not plan.execution_steps:
        raise ValueError("Execution plan must contain at least one execution step.")

    seen_requested = set()
    saw_city_lookup = False
    weather_step_count = 0
    city_lookup_count = 0
    for step in plan.execution_steps:
        if step.tool not in available:
            raise ValueError(f"Execution step uses unsupported tool: {step.tool}")
        if step.tool != "city_to_coords" and step.tool not in user_requested:
            raise ValueError(f"Execution plan added unrequested tools: {step.tool}")
        if step.tool != "city_to_coords":
            seen_requested.add(step.tool)
        if step.tool == "city_to_coords":
            saw_city_lookup = True
            city_lookup_count += 1
        if step.tool == "get_weather":
            weather_step_count += 1
            has_coords = (
                step.args.get("latitude") is not None
                and step.args.get("longitude") is not None
                or plan.location is not None
                and plan.location.latitude is not None
                and plan.location.longitude is not None
            )
            if not has_coords and not saw_city_lookup:
                raise ValueError("Weather execution requires coordinates or a prior city_to_coords step.")

    if city_lookup_count and "get_weather" not in user_requested:
        raise ValueError("city_to_coords is only valid as a dependency for weather requests.")
    if city_lookup_count and (
        coords_in_prompt is not None
        or (
            plan.location is not None
            and plan.location.latitude is not None
            and plan.location.longitude is not None
        )
    ):
        raise ValueError("Execution plan added city_to_coords even though coordinates were already provided.")
    if weather_step_count > 1 or city_lookup_count > 1:
        raise ValueError("Execution plan contains duplicate weather dependency steps.")

    missing_steps = user_requested.difference(seen_requested)
    if missing_steps:
        raise ValueError(f"Execution plan omitted requested tools: {', '.join(sorted(missing_steps))}")
