from __future__ import annotations

import html
from typing import Any, Dict

import requests

from mcp_runtime.registry import mcp
from tools.shared import error_payload, get_json


@mcp.tool()
def random_joke() -> Dict[str, Any]:
    """Return a safe one-line joke via JokeAPI."""
    try:
        data = get_json("https://v2.jokeapi.dev/joke/Any?type=single&safe-mode")
    except requests.RequestException as exc:
        return error_payload("joke", exc)

    return {"joke": data.get("joke", "No joke found.")}


@mcp.tool()
def random_dog() -> Dict[str, Any]:
    """Return a random dog image URL via Dog CEO."""
    try:
        data = get_json("https://dog.ceo/api/breeds/image/random")
    except requests.RequestException as exc:
        return error_payload("dog", exc)

    return {"status": data.get("status"), "image_url": data.get("message")}


@mcp.tool()
def trivia() -> Dict[str, Any]:
    """Return one multiple-choice trivia question via Open Trivia DB."""
    try:
        data = get_json("https://opentdb.com/api.php?amount=1&type=multiple")
    except requests.RequestException as exc:
        return error_payload("trivia", exc)

    results = data.get("results", [])
    if not results:
        return {"error": "trivia request returned no results"}

    question = results[0]
    return {
        "category": html.unescape(question.get("category", "")),
        "difficulty": question.get("difficulty"),
        "question": html.unescape(question.get("question", "")),
        "correct_answer": html.unescape(question.get("correct_answer", "")),
        "incorrect_answers": [
            html.unescape(answer)
            for answer in question.get("incorrect_answers", [])
        ],
    }
