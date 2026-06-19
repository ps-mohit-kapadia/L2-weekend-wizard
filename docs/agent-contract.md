# Weekend Wizard Agent Contract

## Purpose

Weekend Wizard is a bounded weekend-planning agent. It answers supported user requests by combining a local LLM, fixed MCP tools, grounded tool observations, and one reflection pass.

The agent helps with lightweight weekend planning and enrichment using public data:

- current weather
- city-to-coordinate lookup
- book recommendations
- safe jokes
- random dog photos
- trivia questions

## Problem Statement Mapping

Weekend Wizard maps to the problem statement: "What should I do this weekend?"

It can answer direct tool-backed requests, such as:

- "Plan a cozy Saturday in New York with today's weather, 3 mystery book ideas, a joke, and a dog pic."
- "I'm at 40.7128, -74.0060. Give me the current weather, one joke, and a dog photo."
- "Give me one trivia question."

## Input Contract

The agent accepts one natural-language prompt per interaction.

Supported prompt categories:

- weather by coordinates
- weather by city, with city lookup first
- book recommendations by topic
- one safe joke
- one random dog photo
- one trivia question
- combinations of the above in a single weekend-planning request

API prompt size is bounded by `WEEKEND_WIZARD_MAX_PROMPT_CHARS`.

## Output Contract

The core agent returns:

- `answer`: final grounded answer text
- `tool_observations`: structured records of tool calls and serialized payloads
- `used_fallback`: whether the grounded fallback/degraded path was used

Access layers may expose this differently:

- HTTP API returns `answer` and `tool_observations`.
- CLI prints the answer and optionally tool observations.
- Streamlit renders the answer and expandable tool observations.
- A2A should expose the answer as an artifact over the A2A envelope.

## Bounded Scope

Weekend Wizard will:

- use supported tools only when they match the requested work
- produce grounded answers from tool observations
- combine supported categories into a concise weekend plan
- safely report degraded tool outcomes when a requested tool fails
- refuse or avoid unsupported claims, such as completed bookings or purchases

Weekend Wizard will not:

- make reservations or bookings
- take payments or perform purchases
- store personal data
- browse arbitrary websites
- access arbitrary files or shell commands
- provide professional medical, legal, or financial advice
- reveal hidden prompts, secrets, environment values, or internal instructions
- expand tool use beyond the declared supported tools

## Tool Access and Least Privilege

Tools are fixed MCP capabilities. The agent cannot call arbitrary tools.

Allowed tools:

- `city_to_coords`: resolve a city name to coordinates
- `get_weather`: fetch current weather for latitude and longitude
- `book_recs`: fetch book recommendations for a topic and limit
- `random_joke`: fetch one safe joke
- `random_dog`: fetch one random dog image URL
- `trivia`: fetch one trivia question and answer

Tool usage must align with requested work. For example, `city_to_coords` is a dependency for city weather, not a standalone fulfillment of weather.

## Data Handling

The agent uses public APIs for supported tool data.

Data handling rules:

- user prompts are processed for the current interaction
- runtime logs and traces may include prompt length, request flow, tool names, arguments, and bounded payload previews
- tool failure details returned to users are sanitized where needed
- secrets and hidden system/developer instructions must not be returned to users
- no persistent user profile or memory store is maintained by this agent

## Reliability Controls

Reliability controls include:

- bounded ReAct step count
- strict ReAct decision validation
- supported-tool validation
- tool argument normalization
- duplicate successful tool-call prevention
- grounded draft generation from tool observations
- reflection preservation check against required grounded fact IDs
- safe fallback to grounded draft when reflection fails or drifts
- API startup readiness and retry for retryable runtime failures
- API request timeout through `WEEKEND_WIZARD_REQUEST_TIMEOUT`
- tool HTTP retries and timeouts through configuration

## Observability

Weekend Wizard emits human-readable and machine-searchable operational signals.

Signals include:

- API lifecycle logs
- request traces
- LLM call start/completion events
- tool execution start/completion events
- tool duration metadata
- reflection rejection reason codes
- fallback/degraded behavior markers
- semantic log fields such as `event=...` and `reason=...`

## Security and Prompt-Injection Posture

Model output is treated as a proposal, not authority.

Security posture:

- HTTP `/chat` can require `X-API-Key` when `WEEKEND_WIZARD_API_KEY` is configured
- `/health` and `/ready` remain open for operational probes
- prompt size is bounded at the API boundary
- basic in-memory rate limiting protects local/demo use
- ReAct tool names are validated against allowed tools
- reflection cannot override grounded required facts
- unsupported requests should not expand tool scope
- hidden prompts, secrets, and internal instructions must not be exposed

## Access Interfaces

Current access interfaces:

- CLI: `python main.py chat "<prompt>"`
- HTTP API: `/chat`, `/health`, `/ready`
- Streamlit UI: `python main.py streamlit`
- A2A-compatible adapter: `/.well-known/agent.json`, `/a2a/jsonrpc`

## Evaluation Gate

Evaluation artifacts live under `evals/`.

Current eval assets:

- `evals/cases.jsonl`: acceptance cases
- `evals/rubric.md`: pass/fail criteria
- `evals/runner.py`: API-based eval runner
- `evals/report.md`: generated report target

The eval gate checks tool coverage, forbidden tool use, observation bounds, answer completeness, and answer safety markers.

## Known Limitations

- Local LLM latency can be high for multi-step prompts.
- Behavior quality depends on the configured local model.
- API rate limiting is in-memory and intended for local/demo use.
- A2A compatibility is limited to a minimal synchronous text adapter.
- The agent is intentionally limited to supported public-data tools.

## Future Hardening

Planned or possible hardening:

- broader A2A support for streaming, cancellation, task polling, and signed Agent Cards
- external gateway or distributed rate limiting for multi-process deployment
- richer eval reports with historical comparison
- stronger production auth such as OAuth/JWT if deployed beyond local/demo scope
- additional telemetry aggregation for latency and failure trends
