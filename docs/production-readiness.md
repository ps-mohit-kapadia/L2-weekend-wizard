# Production Readiness

Weekend Wizard is a local, bounded agent with CLI, HTTP API, Streamlit, and minimal A2A-compatible access paths. This document summarizes the production-readiness posture without changing the runtime contract.

## Interfaces

- CLI: `python main.py chat "<prompt>"`
- HTTP API: `POST /chat`, `GET /health`, `GET /ready`
- A2A: `GET /.well-known/agent.json`, `POST /a2a/jsonrpc`
- Streamlit: `python main.py streamlit`

## Example Requests

HTTP chat:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/chat `
  -ContentType "application/json" `
  -Body '{"prompt":"Give me one trivia question."}'
```

A2A Agent Card:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/.well-known/agent.json
```

A2A message/send:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/a2a/jsonrpc `
  -ContentType "application/json" `
  -Body '{"jsonrpc":"2.0","id":"req-1","method":"message/send","params":{"message":{"role":"user","parts":[{"kind":"text","text":"Give me one trivia question."}]}}}'
```

When `WEEKEND_WIZARD_API_KEY` is configured, include:

```powershell
-Headers @{"X-API-Key"="<key>"}
```

## Reliability

- Runtime startup validates Ollama, model availability, MCP session readiness, and discovered tools.
- Retryable startup failures are retried without requiring process restart.
- Requests are bounded by `WEEKEND_WIZARD_REQUEST_TIMEOUT`.
- ReAct execution is bounded by step count and supported-tool validation.
- Reflection cannot override required grounded facts; degraded fallback returns grounded content.

## Security

- `/chat` and A2A calls can require `X-API-Key` with `WEEKEND_WIZARD_API_KEY`.
- `/health` and `/ready` remain open for operational probes.
- Prompt size is bounded by `WEEKEND_WIZARD_MAX_PROMPT_CHARS`.
- Local/demo rate limiting is controlled by `WEEKEND_WIZARD_RATE_LIMIT_REQUESTS` and `WEEKEND_WIZARD_RATE_LIMIT_WINDOW_SECONDS`.
- Tools are fixed MCP capabilities; the agent cannot call arbitrary shell, filesystem, browser, or purchase/booking actions.

## Observability

- API lifecycle logs show startup, readiness, and request completion.
- App logs keep the existing human-readable format and add stable key/value fields such as `event`, `interface`, `component`, `status`, `provider`, `model`, `tool_count`, and `correlation_id` where applicable.
- `/ready` exposes provider, model, tools, auth, rate-limit, timeout, trace-logging, and dependency diagnostics.
- Request traces include LLM calls, tool calls, tool durations, observations, fallback state, and final answer length.
- Semantic reason codes explain reflection rejection, fallback, repair, and request-completion decisions.

## Quality Gates

Run unit and integration tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Run evals after starting the API:

```powershell
.\.venv\Scripts\python.exe .\evals\runner.py
```

The eval gate checks required tool coverage, forbidden tool use, observation bounds, answer completeness, and answer safety markers.

## Known Limits

- Local LLM latency can be high for multi-step prompts.
- A2A support is synchronous text-only and does not yet include streaming, polling, cancellation, push notifications, or signed Agent Cards.
- Rate limiting is in-memory and intended for local/demo deployment, not multi-process production gateways.
- Stronger production auth such as OAuth/JWT is future hardening if deployed beyond local/demo use.
