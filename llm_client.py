from __future__ import annotations

"""Local Ollama client helpers for ReAct decisions and reflection JSON outputs."""

import json
import time
from typing import Any, Dict, List, Optional

import requests

from config.config import get_settings
from logger.logging import get_logger
from logger.tracing.request_trace import RequestTrace
from schemas.agent import (
    ReactDecision,
    ReflectionResult,
    validate_react_decision,
    validate_reflection_result,
)

logger = get_logger("llm_client")


def _resolve_model_name(settings: Any, cli_model: Optional[str]) -> str:
    if cli_model:
        return cli_model
    configured_models = getattr(settings, "preferred_models", ())
    if not configured_models:
        raise RuntimeError("No model is configured.")
    return configured_models[0]


def _call_ollama_model(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    json_mode: bool,
    *,
    settings: Any,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    response = requests.post(
        settings.ollama_url,
        json=payload,
        timeout=settings.request_timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["message"]["content"]


def _call_aiplatform_model(
    messages: List[Dict[str, str]],
    model: str,
    temperature: float,
    json_mode: bool,
    *,
    settings: Any,
) -> str:
    if not settings.aiplatform_api_key:
        raise RuntimeError("AIPLATFORM_API_KEY is required for llm_provider=aiplatform.")

    chat_path = settings.aiplatform_chat_path
    if not chat_path.startswith("/"):
        chat_path = "/" + chat_path

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    response = requests.post(
        f"{settings.aiplatform_base_url}{chat_path}",
        json=payload,
        headers={
            "Authorization": f"Bearer {settings.aiplatform_api_key}",
            "Content-Type": "application/json",
        },
        timeout=settings.aiplatform_timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def list_available_models(timeout: int = 5) -> List[str]:
    """Return the models currently reported by the local Ollama runtime."""
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
    *,
    trace: RequestTrace | None = None,
) -> str:
    """Call the configured LLM provider and return raw message content."""
    settings = get_settings()

    if trace is not None:
        trace.add_event(
            "llm_call_started",
            model=model,
            messages_count=len(messages),
        )

    logger.info(
        "Calling %s model %s with %d messages (json_mode=%s, temperature=%s)",
        settings.llm_provider,
        model,
        len(messages),
        json_mode,
        temperature,
    )
    started = time.perf_counter()
    if settings.llm_provider == "ollama":
        content = _call_ollama_model(
            messages,
            model,
            temperature,
            json_mode,
            settings=settings,
        )
    elif settings.llm_provider == "aiplatform":
        content = _call_aiplatform_model(
            messages,
            model,
            temperature,
            json_mode,
            settings=settings,
        )
    else:
        raise RuntimeError(f"Unsupported llm_provider: {settings.llm_provider}")
    duration_ms = int((time.perf_counter() - started) * 1000)
    logger.info("Received %s response for model %s", settings.llm_provider, model)
    if trace is not None:
        trace.add_event(
            "llm_call_completed",
            model=model,
            duration_ms=duration_ms,
            messages_count=len(messages),
        )
    return content


def discover_model(cli_model: Optional[str]) -> str:
    """Resolve the configured model name for the current run."""
    settings = get_settings()
    configured_model = _resolve_model_name(settings, cli_model)
    if settings.llm_provider == "aiplatform":
        return configured_model
    try:
        names = list_available_models(timeout=5)
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach Ollama to validate model '{configured_model}': {exc}"
        ) from exc

    if configured_model not in names:
        raise RuntimeError(
            f"Configured Ollama model is not available: {configured_model}"
        )

    return configured_model


def extract_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object embedded in model output."""
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


def _extract_valid_decision_json(text: str) -> ReactDecision:
    return validate_react_decision(extract_json(text))


def _extract_valid_reflection_json(text: str) -> ReflectionResult:
    return validate_reflection_result(extract_json(text))


def llm_react_json(
    messages: List[Dict[str, str]],
    model: str,
    *,
    allowed_tools: List[str],
    trace: RequestTrace | None = None,
) -> ReactDecision:
    """Return one bounded ReAct decision from Ollama."""
    raw = call_model(messages, model, temperature=0.2, json_mode=True, trace=trace)

    try:
        return _extract_valid_decision_json(raw)
    except Exception:
        logger.warning(
            "Model returned invalid ReAct decision payload; attempting one repair pass"
        )
        allowed_tool_lines = "\n".join(f"- {tool_name}" for tool_name in allowed_tools)
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Return only one valid Weekend Wizard decision JSON object. "
                    'Use either {"thought":"...","action":"tool","tool":"...","args":{}} '
                    'or {"thought":"...","action":"finish","final_answer":"..."}.\n'
                    "You are repairing a prior Weekend Wizard ReAct decision.\n"
                    "Use the original request and prior tool observations below.\n"
                    "Allowed tools:\n"
                    f"{allowed_tool_lines}\n"
                    "Never invent tools.\n"
                    "If prior observations already satisfy the request, return finish.\n"
                    "If one successful random_joke, random_dog, or trivia observation already satisfies the request, return finish."
                ),
            },
            *messages,
            {
                "role": "user",
                "content": (
                    "The previous model output was invalid or unsupported.\n"
                    "Repair it into one valid Weekend Wizard decision JSON object only.\n\n"
                    f"Invalid output:\n{raw}"
                ),
            },
        ]
        repaired = call_model(
            repair_messages, model, temperature=0.0, json_mode=True, trace=trace
        )
        try:
            return _extract_valid_decision_json(repaired)
        except Exception as exc:
            logger.exception("Repair pass did not produce a valid ReAct decision")
            preview = raw.strip().replace("\n", " ")[:200]
            raise ValueError(
                f"Model returned invalid ReAct decision JSON after one repair attempt. Raw output preview: {preview}"
            ) from exc


def llm_reflection_json(
    messages: List[Dict[str, str]],
    model: str,
    *,
    trace: RequestTrace | None = None,
) -> ReflectionResult:
    """Return a JSON reflection payload from Ollama."""
    raw = call_model(messages, model, temperature=0.0, json_mode=True, trace=trace)

    try:
        return _extract_valid_reflection_json(raw)
    except Exception:
        logger.warning(
            "Model returned invalid reflection payload; attempting one repair pass"
        )
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
        repaired = call_model(
            repair_messages, model, temperature=0.0, json_mode=True, trace=trace
        )
        try:
            return _extract_valid_reflection_json(repaired)
        except Exception as exc:
            logger.exception("Repair pass did not produce a valid reflection payload")
            preview = raw.strip().replace("\n", " ")[:200]
            raise ValueError(
                f"Model returned invalid reflection JSON after one repair attempt. Raw output preview: {preview}"
            ) from exc
