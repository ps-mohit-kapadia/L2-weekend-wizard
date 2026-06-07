from __future__ import annotations

import html
from typing import Any

import requests

from mcp_runtime.registry import mcp
from schemas.tools import DogResult, JokeResult, ToolError, TriviaResult
from tools.shared import error_payload, get_json


@mcp.tool()
def random_joke() -> JokeResult | ToolError:
    """Return a safe one-line joke via JokeAPI."""
    try:
        data = get_json("https://v2.jokeapi.dev/joke/Any?type=single&safe-mode")
    except requests.RequestException as exc:
        return error_payload("joke", exc)

    return JokeResult(joke=data.get("joke", "No joke found."))


@mcp.tool()
def random_dog() -> DogResult | ToolError:
    """Return a random dog image URL via Dog CEO."""
    try:
        data = get_json("https://dog.ceo/api/breeds/image/random")
    except requests.RequestException as exc:
        return error_payload("dog", exc)

    return DogResult(status=data.get("status"), image_url=data.get("message"))


@mcp.tool()
def trivia() -> TriviaResult | ToolError:
    """Return one multiple-choice trivia question via Open Trivia DB."""
    try:
        data = get_json("https://opentdb.com/api.php?amount=1&type=multiple")
    except requests.RequestException as exc:
        return error_payload("trivia", exc)

    results = data.get("results", [])
    if not results:
        return ToolError(error="trivia request returned no results")

    question = results[0]
    return TriviaResult(
        category=html.unescape(question.get("category", "")),
        difficulty=question.get("difficulty"),
        question=html.unescape(question.get("question", "")),
        correct_answer=html.unescape(question.get("correct_answer", "")),
        incorrect_answers=[
            html.unescape(answer)
            for answer in question.get("incorrect_answers", [])
        ],
    )
