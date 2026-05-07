from __future__ import annotations

"""Prompt helpers for the first Claude Agent SDK Weekend Wizard slice."""


def build_system_prompt() -> str:
    """Return the system prompt for the first SDK slice.

    This slice intentionally supports only the smallest joke-focused path needed
    to prove the Claude Agent SDK integration while leaving the original L2
    implementation intact.
    """
    return (
        "You are Weekend Wizard, an upbeat weekend helper.\n"
        "For this first Claude Agent SDK slice, only handle joke-style requests.\n"
        "Use the provided random_joke tool when the user asks for a joke.\n"
        "After you have one successful joke result, answer briefly and finish.\n"
        "Do not invent tools or unsupported capabilities."
    )
