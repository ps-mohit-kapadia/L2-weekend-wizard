from __future__ import annotations

"""Local Ollama client helpers for JSON-oriented agent decisions."""

import json
import os
from typing import Any, Dict, List, Optional

import requests

from config.config import get_settings


def list_available_models(timeout: int = 5) -> List[str]:
    """Return the models currently reported by the local Ollama runtime.

    Args:
        timeout: Request timeout in seconds for the Ollama tags endpoint.

    Returns:
        The available Ollama model names.

    Raises:
        requests.RequestException: If the Ollama tags endpoint cannot be reached.
    """
    response = requests.get(
        get_settings().llm.ollama_url.replace("/api/chat", "/api/tags"),
        timeout=timeout,
    )
    response.raise_for_status()
    models = response.json().get("models", [])
    return [model.get("name") for model in models if model.get("name")]


def call_model(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    json_mode: bool = False,
) -> str:
    """Call the local Ollama chat endpoint and return raw message content.

    Args:
        messages: Chat messages to send to the model.
        model: Ollama model name to invoke.
        temperature: Sampling temperature for the request.
        json_mode: Whether to request JSON-formatted model output.

    Returns:
        The raw assistant message content returned by Ollama.

    Raises:
        requests.RequestException: If the HTTP request to Ollama fails.
    """
    settings = get_settings()
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        settings.llm.ollama_url,
        json=payload,
        timeout=90,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def discover_model(cli_model: Optional[str]) -> str:
    """Resolve the Ollama model name for the current run.

    Args:
        cli_model: Optional model override provided on the command line.

    Returns:
        The chosen Ollama model name.
    """
    if cli_model:
        return cli_model

    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model

    try:
        names = list_available_models(timeout=5)
        for name in get_settings().llm.preferred_models:
            if name in names:
                return name
        if names:
            return names[0]
    except requests.RequestException:
        pass

    return "mistral:7b"


def extract_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object embedded in model output.

    Args:
        text: Raw model output that may contain surrounding text.

    Returns:
        The first JSON object parsed from the text.

    Raises:
        json.JSONDecodeError: If no JSON object can be recovered from the text.
    """
    text = text.strip()
    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("No JSON object found", text, 0)


def llm_json(
    messages: List[Dict[str, str]],
    model: str,
) -> Dict[str, Any]:
    """Return a JSON decision from Ollama.

    Args:
        messages: Chat history for the current decision request.
        model: Ollama model name to invoke.

    Returns:
        A JSON decision payload for the agent loop.

    Raises:
        requests.RequestException: If the Ollama request fails in normal mode.
        json.JSONDecodeError: If model output cannot be repaired into valid JSON.
    """
    raw = call_model(messages, model, temperature=0.2, json_mode=True)

    try:
        return extract_json(raw)
    except json.JSONDecodeError:
        repair_messages = [
            {"role": "system", "content": "Return only valid JSON that preserves the user's intent."},
            {"role": "user", "content": raw},
        ]
        repaired = call_model(repair_messages, model, temperature=0.0, json_mode=True)
        return extract_json(repaired)
