# Weekend Wizard (L2 ReAct Agent Project)

A lightweight local AI agent that helps answer:

**"What should I do this weekend?"**

Weekend Wizard combines a local LLM, MCP-exposed tools, and a bounded ReAct loop to build short, grounded weekend suggestions using real public data.

The system can pull:

- current weather
- book recommendations
- a safe joke
- a random dog photo
- optional trivia

This branch is intentionally aligned to the original L2 assignment:

**perceive -> decide -> act -> observe -> repeat -> reflect once**

The agent decides one next step at a time, calls MCP tools when needed, observes the result, and stops when it has enough information to answer.

For the parallel Level 3 Claude Agent SDK migration path, see [docs/sdk-migration.md](docs/sdk-migration.md).

For access interfaces, runtime controls, examples, and quality gates, see [docs/production-readiness.md](docs/production-readiness.md).

## L3 Claude SDK Path

This branch also includes a parallel Claude Agent SDK implementation under `claude_sdk_agent/`.

Migration details live in [docs/sdk-migration.md](docs/sdk-migration.md).

Useful SDK dry-run commands:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_claude_sdk_agent.py --dry-run --prompt "Tell me a joke."
.\.venv\Scripts\python.exe .\scripts\smoke_claude_sdk_agent.py --dry-run
```

Notes:

- the original L2 ReAct implementation and entrypoints below remain intact
- the SDK path is implementation-complete; live verification requires Claude API access
- the live runner requires valid Claude SDK / Claude Code authentication available in the environment
- the live smoke harness currently pre-checks `ANTHROPIC_API_KEY`
- alternative company Claude Code auth paths are still pending clarification

---

## Architecture Overview

Weekend Wizard follows a bounded ReAct-style architecture:

1. the user prompt enters the runtime
2. the LLM decides the next action
3. the orchestrator either executes one tool or finishes
4. tool observations are appended to history
5. the loop repeats until the model finishes or the step budget is reached
6. one reflection pass lightly corrects the final answer

```mermaid
flowchart TD
    A["User Prompt"] --> B["Streamlit UI"]
    A --> CLI["CLI chat"]
    A --> A2A["A2A message/send"]
    B --> C["FastAPI /chat"]
    CLI --> D["application/service.py"]
    A2A --> C
    C --> D["application/service.py"]
    D --> E["agent/orchestrator.py"]
    E --> F["ReAct Prompt"]
    F --> G["LLM ReAct Decision"]
    G --> H{"Action?"}
    H -->|tool| I["Normalize Tool Args"]
    I --> J["MCP Runtime Client"]
    J --> K["MCP Server"]
    K --> L["Tool Modules"]
    L --> M["External APIs"]
    M --> N["Tool Observation"]
    N --> E
    H -->|finish| O["Grounded Draft"]
    O --> P["Reflection Prompt"]
    P --> Q["LLM Reflection"]
    Q --> R["Final Grounded Answer"]
    R --> C
    R --> B
    R --> CLI
    R --> A2A
    
    subgraph "LLM Provider"
        G --> OLLAMA["Ollama (local)"]
        G --> AIPLATFORM["AI Platform (hosted)"]
        Q --> OLLAMA
        Q --> AIPLATFORM
    end
```

### Execution Flow

1. Streamlit, CLI, or A2A sends the user prompt into the same runtime.
2. The ReAct LLM produces one bounded JSON decision.
3. If the decision is a tool call, the orchestrator validates and executes it through MCP.
4. The tool observation is added to conversation history.
5. Steps 2 to 4 repeat until the model chooses `finish` or the max step budget is reached.
6. Grounding builds a draft answer from real observations.
7. The reflection LLM performs one lightweight correction pass.
8. The final grounded answer is returned through the active access layer.

---

## What It Does

Weekend Wizard supports prompts like:

- "Plan a cozy Saturday in New York with today's weather, 3 mystery book ideas, a joke, and a dog pic."
- "I'm at 40.7128, -74.0060. Give me the weather, one joke, and a dog photo."
- "Give me one trivia question."

Supported tool-backed capabilities:

- weather via Open-Meteo
- city-to-coordinates lookup via Open-Meteo geocoding
- book recommendations via Open Library
- a safe one-liner joke via JokeAPI
- a random dog photo URL via Dog CEO
- optional trivia via Open Trivia DB

`trivia` is intentionally supported as an explicit request, not as automatic enrichment for unrelated prompts.

---

## Project Structure

```text
weekend-wizard/
|- main.py
|- api.py
|- streamlit_app.py
|- llm_client.py
|- mcp_server.py
|- requirements.txt
|- README.md
|
|- claude_sdk_agent/        # parallel L3 Claude SDK path
|  |- config.py
|  |- prompts.py
|  |- runner.py
|  |- smoke.py
|  |- tools.py
|
|- application/
|  |- service.py
|
|- agent/
|  |- grounding.py
|  |- orchestrator.py
|  |- prompts.py
|  |- policies/
|     |- guardrails.py
|
|- config/
|  |- config.py
|
|- logger/
|  |- logging.py
|
|- mcp_runtime/
|  |- client.py
|  |- registry.py
|
|- schemas/
|  |- agent.py
|  |- a2a.py
|  |- api.py
|  |- tools.py
|
|- docs/
|  |- a2a.md
|  |- agent-contract.md
|  |- production-readiness.md
|  |- sdk-migration.md
|
|- evals/
|  |- cases.jsonl
|  |- report.md
|  |- rubric.md
|  |- runner.py
|
|- tools/
|  |- books.py
|  |- entertainment.py
|  |- geo.py
|  |- shared.py
|  |- weather.py
|
|- scripts/                 # parallel L3 Claude SDK runner/smoke entrypoints
|  |- run_claude_sdk_agent.py
|  |- smoke_claude_sdk_agent.py
|
|- tests/
|  |- smoke/
|  |  |- smoke_test.py
|  |- integration/
|  |- unit/
|     |- test_a2a.py
```

---

## Components

### 1. Streamlit UI (`streamlit_app.py`)

The Streamlit app provides the local interactive interface.

Responsibilities:

- collect prompts
- send requests to the backend
- render the final answer and tool observations

---

### 2. FastAPI Backend (`api.py`)

This module exposes the runtime over HTTP.

Responsibilities:

- expose `/chat`, `/health`, `/ready`, `/.well-known/agent.json`, and `/a2a/jsonrpc`
- own the shared runtime lifecycle
- return structured responses using the configured runtime model

---

### 3. Runtime Service (`application/service.py`)

This layer owns shared runtime state for each application session.

Responsibilities:

- initialize the MCP-backed runtime
- resolve the configured model and discovered tools
- create per-request interaction context
- dispatch interactions through the orchestrator

---

### 4. Orchestrator (`agent/orchestrator.py`)

This is the core runtime brain.

Responsibilities:

- build one bounded ReAct prompt per step
- call the ReAct LLM for the next decision
- validate the decision and supported tool use
- normalize tool arguments before execution
- execute MCP tools one step at a time
- record `ToolObservation`s
- build grounded draft answers
- run one reflection pass
- return bounded failure behavior if the loop becomes unreliable

---

### 5. Prompts (`agent/prompts.py`)

This module builds prompt payloads for:

- each bounded ReAct decision
- the one-shot reflection step

---

### 6. Grounding (`agent/grounding.py`)

This module keeps the answer tied to observed tool output.

Responsibilities:

- parse serialized tool payloads
- normalize tool outputs
- compose grounded answers from observations

---

### 7. MCP Runtime (`mcp_runtime/client.py`)

This is the tool execution boundary.

Responsibilities:

- connect to the MCP server
- discover tools
- invoke tools with structured arguments
- return results to the orchestrator

---

### 8. LLM Client (`llm_client.py`)

This module manages LLM interaction through either Ollama or AI Platform.

Responsibilities:

- discover available local models (Ollama only)
- call the bounded ReAct LLM
- call the reflection LLM
- perform one repair attempt for invalid ReAct or reflection JSON

Supported providers:

- **Ollama**: local runtime for privacy-friendly execution
- **AI Platform**: company AI platform endpoint for hosted model access

---

## Running the Project

### 1. Install dependencies

```powershell
cd <path-to-weekend-wizard>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

---

### 2. Configure LLM Provider

Choose your LLM provider based on your needs:

**Option A: Ollama (local)**

Ensure Ollama is running with a local chat model:

```powershell
ollama list
```

If needed:

```powershell
ollama pull llama3.1:8b
```

Set in `.env`:

```env
LLM_PROVIDER=ollama
MODEL=llama3.1:8b
```

**Option B: AI Platform (hosted)**

Set in `.env`:

```env
LLM_PROVIDER=aiplatform
AIPLATFORM_API_KEY=<secret-key>:<public-key>
MODEL=<platform-model-identifier>
```

See the [LLM Provider Configuration](#llm-provider-configuration) section for detailed guidance.

---

### 3. Start the API

```powershell
python .\main.py api
```

---

### 4. Start Streamlit

```powershell
python .\main.py streamlit
```

CLI chat is also available:

```powershell
python .\main.py chat "Give me one trivia question."
```

Useful URLs:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/ready`
- `http://127.0.0.1:8000/.well-known/agent.json`
- `http://127.0.0.1:8000/docs`

---

### 5. Optional: run the MCP server directly

```powershell
python .\main.py mcp-server
```

---

### 6. Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

### 7. Run the smoke test

```powershell
.\.venv\Scripts\python.exe .\tests\smoke\smoke_test.py --prompt "Tell me a joke."
```

### 8. Run evals

Start the API first, then run:

```powershell
.\.venv\Scripts\python.exe .\evals\runner.py
```

---

## LLM Provider Configuration

Weekend Wizard supports two LLM providers:

### Ollama (Local)

Ollama provides local LLM execution for privacy-friendly, cost-free inference.

**When to use Ollama:**

- Local development and testing
- Privacy-sensitive workloads
- No external API costs
- Reproducible runtime environment

**Ollama configuration:**

```env
LLM_PROVIDER=ollama
OLLAMA_URL=http://127.0.0.1:11434/api/chat
MODEL=llama3.1:8b
```

**Ollama setup:**

```powershell
# List available models
ollama list

# Pull a model if needed
ollama pull llama3.1:8b
```

### AI Platform (Hosted)

AI Platform provides access to company-hosted models through a centralized endpoint.

**When to use AI Platform:**

- Production or shared environments
- Stronger model capabilities
- Centralized model management
- When local GPU resources are limited

**AI Platform configuration:**

```env
LLM_PROVIDER=aiplatform
AIPLATFORM_API_KEY=<secret-key>:<public-key>
AIPLATFORM_BASE_URL=https://aiapidev.3ecompany.com
AIPLATFORM_TIMEOUT=120
AIPLATFORM_CHAT_PATH=/v1/chat/completions
MODEL=<platform-model-identifier>
```

**AI Platform notes:**

- `AIPLATFORM_API_KEY` is required when `LLM_PROVIDER=aiplatform`
- The API key format is `<secret-key>:<public-key>`
- Model discovery is skipped for AI Platform (model validation is provider-side)
- Timeout defaults to 120 seconds; increase for slower models

### Provider Selection Guidance

| Factor | Ollama | AI Platform |
|--------|--------|-------------|
| Privacy | Local execution | Hosted endpoint |
| Cost | Free | Company platform |
| Model strength | Depends on local hardware | Access to stronger models |
| Setup complexity | Requires Ollama installation | Requires API key |
| Network dependency | None | Requires network access |
| Model discovery | Automatic via Ollama | Provider-side |

---

## Configuration

Configuration is managed through environment variables and repo config.

Example values for Ollama:

```env
LLM_PROVIDER=ollama
OLLAMA_URL=http://127.0.0.1:11434/api/chat
MODEL=llama3.1:8b

WEEKEND_WIZARD_REQUEST_TIMEOUT=1200
WEEKEND_WIZARD_HTTP_MAX_RETRIES=2
WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS=0.5

WEEKEND_WIZARD_LOG_LEVEL=WARNING  # Use INFO for debugging, WARNING for production
WEEKEND_WIZARD_API_URL=http://127.0.0.1:8000
WEEKEND_WIZARD_API_KEY=
WEEKEND_WIZARD_MAX_PROMPT_CHARS=4000
WEEKEND_WIZARD_RATE_LIMIT_REQUESTS=20
WEEKEND_WIZARD_RATE_LIMIT_WINDOW_SECONDS=60
```

Example values for AI Platform:

```env
LLM_PROVIDER=aiplatform
AIPLATFORM_API_KEY=<secret-key>:<public-key>
AIPLATFORM_BASE_URL=https://aiapidev.3ecompany.com
AIPLATFORM_TIMEOUT=120
AIPLATFORM_CHAT_PATH=/v1/chat/completions
MODEL=<platform-model-identifier>

WEEKEND_WIZARD_REQUEST_TIMEOUT=600
WEEKEND_WIZARD_HTTP_MAX_RETRIES=2
WEEKEND_WIZARD_HTTP_RETRY_BACKOFF_SECONDS=0.5

WEEKEND_WIZARD_LOG_LEVEL=WARNING  # Use INFO for debugging, WARNING for production
WEEKEND_WIZARD_API_URL=http://127.0.0.1:8000
WEEKEND_WIZARD_API_KEY=
WEEKEND_WIZARD_MAX_PROMPT_CHARS=4000
WEEKEND_WIZARD_RATE_LIMIT_REQUESTS=20
WEEKEND_WIZARD_RATE_LIMIT_WINDOW_SECONDS=60
```

Notes:

- the active runtime model is configured in [config/config.py](config/config.py)
- `WEEKEND_WIZARD_API_URL` controls where Streamlit sends requests
- `WEEKEND_WIZARD_REQUEST_TIMEOUT` is especially relevant for slower local Ollama runs
- `WEEKEND_WIZARD_API_KEY` enables `X-API-Key` protection for `/chat` and A2A calls when set
- prompt size and rate-limit settings protect the local/demo API boundary
- the default log level is `WARNING` for production use; set `WEEKEND_WIZARD_LOG_LEVEL=INFO` in your `.env` file for more verbose debugging output
- see `.env.example` for the full configuration template

---

## Health Check

### `/health`

Confirms that the API process is alive.

### `/ready`

Checks whether the backend runtime is actually usable, including:

- configured provider and provider reachability
- configured model availability
- MCP session readiness
- discovered tools
- auth, rate-limit, request-timeout, and trace-logging diagnostics

---

## Design Decisions

### Dual LLM Provider Support

Weekend Wizard supports both local Ollama and hosted AI Platform providers.

**Ollama advantages:**

- privacy-friendly local execution
- no external LLM API cost
- reproducible runtime environment
- strong alignment with the training project goal

**AI Platform advantages:**

- access to stronger hosted models
- centralized model management
- no local GPU requirements
- production-ready endpoint

The provider is selected via `LLM_PROVIDER` environment variable.

---

### Minimal bounded ReAct loop

The system uses:

**LLM decides one next action -> system executes -> LLM decides again**

with:

- a strict JSON decision schema
- a max step budget
- deterministic tool execution
- one reflection pass at the end

This gives the project:

- clear ReAct-style behavior for the assignment
- bounded runtime behavior
- easy-to-follow tool traces
- cleaner debugging than an unrestricted free-form loop

---

### MCP as the tool boundary

Tools are exposed and invoked through MCP rather than embedded directly into prompt logic.

Advantages:

- clear system boundaries
- structured tool invocation
- easier testing
- cleaner separation between agent logic and external APIs

---

### One reflection pass only

Reflection is deliberately limited to a single pass.

This keeps the final answer:

- grounded
- concise
- less prone to looping or re-planning

---

## Sample Prompts

```text
Plan a cozy Saturday in New York with today's weather, 3 mystery book ideas, a joke, and a dog pic.
```

```text
I'm at 40.7128, -74.0060. Give me the current weather, one joke, and a dog photo.
```

```text
Give me one trivia question.
```

---

## Known Limitations

This implementation is a strong local L2 prototype, but a few practical tradeoffs remain.

Limitations:

- local-model latency is still noticeable, especially across multiple ReAct steps and reflection
- runtime quality is model-sensitive, with stronger local models producing better step decisions at the cost of slower responses
- A2A support is a minimal synchronous text adapter, not streaming or long-running task orchestration
- API rate limiting is in-memory and intended for local/demo use
- the system is intentionally bounded to the supported tool-backed flows rather than open-ended general agent behavior

These tradeoffs prioritize:

- clarity
- assignment alignment
- grounded behavior
- explainable execution

over unrestricted flexibility.

---

## Summary

Weekend Wizard demonstrates a local MCP-backed ReAct-style agent that:

- uses Ollama or AI Platform to decide one next action at a time
- executes tool calls deterministically through MCP
- observes real tool output before deciding what to do next
- runs one lightweight reflection pass before replying
- exposes CLI, HTTP API, Streamlit, and minimal A2A-compatible access paths
- includes repeatable tests and evals for regression confidence

The architecture is intentionally modular, bounded, and teachable, making it a strong L2-style capstone implementation for the original assignment.
