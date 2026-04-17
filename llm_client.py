from __future__ import annotations

"""Local Ollama client helpers for planner and reflection JSON outputs."""

import json
from typing import Any, Dict, List, Optional

import requests

from config.config import get_settings
from logger.logging import get_logger
from schemas.agent import validate_execution_plan, validate_reflection_result

MODEL_REQUEST_TIMEOUT_SECONDS = 600
logger = get_logger("llm_client")


def list_available_models(timeout: int = 5) -> List[str]:
    """Return the models currently reported by the local Ollama runtime.

    Args:
        timeout: Request timeout in seconds for the Ollama tags endpoint.

    Returns:
        The available Ollama model names.

    Raises:
        requests.RequestException: If the Ollama tags endpoint cannot be reached.
    """
    logger.info("Requesting available Ollama models with timeout %ss", timeout)
    response = requests.get(
        get_settings().ollama_url.replace("/api/chat", "/api/tags"),
        timeout=timeout,
    )
    response.raise_for_status()
    models = response.json().get("models", [])
    names = [model.get("name") for model in models if model.get("name")]
    logger.info("Discovered %d Ollama models", len(names))
    return names


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

    logger.info(
        "Calling Ollama model %s with %d messages (json_mode=%s, temperature=%s)",
        model,
        len(messages),
        json_mode,
        temperature,
    )
    response = requests.post(
        settings.ollama_url,
        json=payload,
        timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    logger.info("Received Ollama response for model %s", model)
    return data["message"]["content"]


def discover_model(cli_model: Optional[str]) -> str:
    """Resolve the Ollama model name for the current run.

    Args:
        cli_model: Legacy parameter retained for compatibility. The configured model is used.

    Returns:
        The chosen Ollama model name.

    Raises:
        RuntimeError: If no configured model exists or Ollama does not expose it.
    """
    settings = get_settings()
    configured_models = settings.preferred_models
    if not configured_models:
        raise RuntimeError("No Ollama model is configured.")

    configured_model = configured_models[0]
    try:
        names = list_available_models(timeout=5)
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach Ollama to validate model '{configured_model}': {exc}") from exc

    if configured_model not in names:
        raise RuntimeError(f"Configured Ollama model is not available: {configured_model}")

    return configured_model


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


def _validate_plan_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a parsed JSON payload matches the execution plan contract."""
    validate_execution_plan(payload)
    return payload


def _extract_valid_plan_json(text: str) -> Dict[str, Any]:
    """Extract and validate one execution plan payload from raw model output."""
    return _validate_plan_payload(extract_json(text))


def _validate_reflection_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure a parsed JSON payload matches the reflection contract."""
    validate_reflection_result(payload)
    return payload


def _extract_valid_reflection_json(text: str) -> Dict[str, Any]:
    """Extract and validate one reflection payload from raw model output."""
    return _validate_reflection_payload(extract_json(text))


def llm_plan_json(
    messages: List[Dict[str, str]],
    model: str,
) -> Dict[str, Any]:
    """Return a JSON execution plan from Ollama.

    Args:
        messages: Chat history for the current decision request.
        model: Ollama model name to invoke.

    Returns:
        A JSON execution plan for the agent loop.

    Raises:
        requests.RequestException: If the Ollama request fails in normal mode.
        ValueError: If model output cannot be repaired into a valid agent decision.
    """
    raw = call_model(messages, model, temperature=0.2, json_mode=True)

    try:
        return _extract_valid_plan_json(raw)
    except Exception:
        logger.warning("Model returned invalid plan payload; attempting one repair pass")
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Return only one valid execution plan JSON object. "
                    "Preserve the user's intent and use only supported tools."
                ),
            },
            {"role": "user", "content": raw},
        ]
        repaired = call_model(repair_messages, model, temperature=0.0, json_mode=True)
        try:
            return _extract_valid_plan_json(repaired)
        except Exception as exc:
            logger.exception("Repair pass did not produce a valid execution plan")
            preview = raw.strip().replace("\n", " ")[:200]
            raise ValueError(
                f"Model returned invalid execution plan JSON after one repair attempt. Raw output preview: {preview}"
            ) from exc


def llm_reflection_json(
    messages: List[Dict[str, str]],
    model: str,
) -> Dict[str, Any]:
    """Return a JSON reflection payload from Ollama."""
    raw = call_model(messages, model, temperature=0.0, json_mode=True)

    try:
        return _extract_valid_reflection_json(raw)
    except Exception:
        logger.warning("Model returned invalid reflection payload; attempting one repair pass")
        repair_messages = [
            {
                "role": "system",
                "content": (
                    'Return only one valid JSON object in the shape {"answer":"..."} '
                    "and keep it grounded in the provided observations."
                ),
            },
            {"role": "user", "content": raw},
        ]
        repaired = call_model(repair_messages, model, temperature=0.0, json_mode=True)
        try:
            return _extract_valid_reflection_json(repaired)
        except Exception as exc:
            logger.exception("Repair pass did not produce a valid reflection payload")
            preview = raw.strip().replace("\n", " ")[:200]
            raise ValueError(
                f"Model returned invalid reflection JSON after one repair attempt. Raw output preview: {preview}"
            ) from exc
