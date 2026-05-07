# Claude Agent SDK Migration

This branch keeps the original Level 2 Weekend Wizard ReAct implementation intact and adds a new parallel Claude Agent SDK implementation.

## Why the SDK version exists

The Level 3 requirement is to refactor the Level 2 project into a Claude Agent SDK-based implementation without destroying the original baseline.

This repo now has two paths:

- the original Level 2 ReAct implementation
- a new Claude Agent SDK path under `claude_sdk_agent/`

The goal is to compare and evolve the same Weekend Wizard workflow through a different agent runtime model while keeping the original assignment implementation runnable.

## What changed and why

The original L2 files and entrypoints are still present and still serve as the baseline implementation.

The new SDK path adds:

- `claude_sdk_agent/`
- `scripts/run_claude_sdk_agent.py`
- `scripts/smoke_claude_sdk_agent.py`

The SDK version exists so the project can:

- use a Claude Agent SDK-driven agent loop
- expose an explicit structured toolset
- keep configuration deterministic
- support a separate Level 3 smoke path

The SDK path is intentionally separate so it does not rewrite the original L2 architecture in place.

## What stays the same

- Weekend Wizard remains the same product concept
- the same domain tools are reused where possible
- the original L2 ReAct implementation remains runnable

## What is different

- L2 uses a local Ollama-powered bounded ReAct loop
- L3 uses the Claude Agent SDK path under `claude_sdk_agent/`
- the SDK path currently uses in-process structured SDK tools rather than the original MCP subprocess runtime

## Running the original L2 version

Install dependencies:

```powershell
cd "C:\Users\MohitKapadiya\Desktop\New folder\genai\L2_agents\weekend-wizard"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\requirements.txt
```

Start the L2 API:

```powershell
python .\main.py api
```

Start the L2 Streamlit app:

```powershell
python .\main.py streamlit
```

Run the original L2 smoke test:

```powershell
.\.venv\Scripts\python.exe .\tests\smoke\smoke_test.py --prompt "Tell me a joke."
```

## Running the Claude SDK version

The SDK path uses its own runner script and does not change the L2 entrypoints.

Dry-run the SDK runner without calling Claude:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_claude_sdk_agent.py --dry-run --prompt "Tell me a joke."
```

Dry-run the SDK smoke harness:

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_claude_sdk_agent.py --dry-run
```

Attempt a live SDK run:

```powershell
.\.venv\Scripts\python.exe .\scripts\run_claude_sdk_agent.py --prompt "Tell me a joke."
```

Attempt a live SDK smoke run:

```powershell
.\.venv\Scripts\python.exe .\scripts\smoke_claude_sdk_agent.py
```

Live SDK runs require Claude authentication such as `ANTHROPIC_API_KEY`.

Do not commit real API keys into the repository.

## Tool permissions and assumptions

The current SDK implementation exposes only the explicit Weekend Wizard toolset:

- `random_joke`
- `random_dog`
- `trivia`
- `book_recs`
- `city_to_coords`
- `get_weather`

The SDK path is intentionally explicit and does not expose arbitrary extra tools.

The current SDK runner assumes:

- a Claude SDK installation is present in the repo virtual environment
- any live Claude run has valid authentication in the environment
- dry-run mode should work without Claude authentication

## Context management

The original L2 ReAct path manages its own bounded conversation and tool-observation history inside the orchestrator.

The current SDK path:

- builds a fresh agent run per invocation
- provides a deterministic system prompt
- exposes a fixed explicit tool allowlist
- does not currently persist long-lived conversation state between separate script invocations

What is effectively persisted today:

- static SDK configuration
- explicit tool definitions
- scenario definitions in the smoke harness

What is discarded between separate runs:

- prior prompt history
- prior tool results
- prior Claude session state unless a future SDK session persistence layer is added

## Known limitations

- The original L2 implementation is still the more fully exercised runtime path.
- The SDK path currently focuses on structure, tool wiring, and offline verification.
- `--dry-run` verifies scenario/config shape only. It does not verify live model behavior.
- Live Claude SDK behavior has not yet been verified on this branch unless a later run is explicitly approved and executed.
- The SDK path does not yet expose its own API surface.
- The SDK path does not yet replace the original L2 smoke test or runtime.

## Current recommendation

Use the L2 path as the stable baseline.

Use the SDK path to incrementally build and verify the Level 3 implementation without destabilizing the original Level 2 project.
