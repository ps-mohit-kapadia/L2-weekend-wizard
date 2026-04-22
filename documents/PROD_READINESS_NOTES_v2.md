# Production Readiness Notes v2

## Purpose

This document updates the production-readiness learning story for the `prod-readiness` branch after the latest work on:

- local operator workflow
- evaluation
- access control
- local `.env` configuration
- observability modes

The goal is still not to claim that Weekend Wizard is fully production-ready. The goal is to describe, as accurately as possible, what has been improved, what has been measured, and what still blocks trusted customer-facing deployment.

---

## What Changed On This Branch

### 1. Local operator workflow was simplified

The repo now exposes one operator-facing startup entrypoint:

```powershell
python .\scripts\dev_up.py check
python .\scripts\dev_up.py api
python .\scripts\dev_up.py streamlit
```

This improves local deployability by making the runtime flow:

- explicit
- repeatable
- self-checking

The `check` subcommand validates:

- configured model resolution
- Ollama reachability
- MCP runtime startup
- tool discovery

### 2. Evaluation was added as a first-class repo concern

The repo now includes an `evaluations/` folder with:

- `cases.json`
- `run_evaluations.py`

This provides a lightweight contract-evaluation harness for supported prompts.

The current evaluation checks:

- required tools are present
- forbidden tools are absent
- a minimum number of tool observations is returned
- the final answer is non-empty

### 3. Evaluation failure handling was hardened

The evaluation runner now:

- records per-case failures
- records per-case durations
- converts timeouts into scored failures
- prints a final summary instead of aborting on the first timeout

This makes the evaluation layer usable for real baseline comparison.

### 4. Shared-key access control was added

The API now protects `/ready` and `/chat` behind a shared API key configured through:

- `WEEKEND_WIZARD_API_KEY`

This is not full production authentication or authorization, but it is a meaningful first customer-facing access boundary for this repo size.

### 5. Local environment configuration was improved

The repo now supports loading a local `.env` file, with `.env.example` as the committed template.

This improves:

- local reproducibility
- setup clarity
- secret/config hygiene

while keeping real local values out of git.

### 6. Observability modes were added

The repo now supports:

- `local`
- `staging`
- `production`

through:

- `WEEKEND_WIZARD_OBSERVABILITY_MODE`

The current implementation preserves readable developer logs in `local`, and enables request correlation plus structured phase telemetry in both `staging` and `production`.

Observed telemetry now includes:

- request IDs
- auth acceptance/rejection events
- planner timing
- per-tool timing
- tool-phase timing
- reflection timing
- end-to-end request timing

This means the branch now has a real observability foundation rather than only ad hoc descriptive logs.

---

## Current Measured Baseline

### Baseline model

Current serious baseline:

- `llama3.1:8b`

Rejected serving candidate in this environment:

- `qwen3.5:4b`

### Evaluation result with `llama3.1:8b`

Observed result:

- `5/6` cases passed
- `1/6` case failed by timeout

Passing cases included:

- weather by city
- weather by coordinates plus joke and dog photo
- trivia only
- joke only
- dog photo only

Failing case:

- full weekend planning prompt
- timed out at `180s`

### Evaluation result with `qwen3.5:4b`

Observed result:

- `0/6` cases passed
- all cases timed out at `180s`

Conclusion:

- `qwen3.5:4b` is not a viable serving model for this repo and machine setup

---

## Main Production-Readiness Findings

### 1. Correctness is improving faster than performance

The current architecture is now reasonably stable for supported prompt types.

However, production-style user experience is still blocked by:

- very high request latency
- poor responsiveness on heavier prompt shapes

### 2. The dominant bottleneck is local model inference

A code-path scan did not reveal enough wasteful Python work to explain `100s+` request times.

Some inefficiencies do exist, but they are secondary:

- grounded answer composition currently happens twice in the finalization path
- planner and reflection each allow one repair pass, which can multiply LLM latency
- readiness polling still performs extra model-list checks

These are real cleanup candidates, but they do not explain the full runtime cost.

The main bottleneck is:

- planner inference
- reflection inference
- CPU-only local model execution

### 3. Hardware constraints matter materially

This repo is currently being evaluated on an office laptop without a GPU.

That environment makes the local model tradeoff very clear:

- stronger local models are more reliable
- stronger local models are also much slower

In this setup, the system is primarily limited by:

- CPU-bound local LLM inference

not by normal application logic.

### 4. Observability now gives phase-level evidence

Recent staging and production-mode runs confirmed that:

- planner is a major latency source
- reflection is a major latency source
- tool execution is comparatively cheap

Representative request patterns showed:

- planner phases in the tens to hundreds of seconds
- tool phases in low single-digit seconds
- reflection phases also in the tens to hundreds of seconds

This means the repo now has measured evidence, not just intuition, about where request time is going.

### 5. Access control exists, but full security maturity does not

The shared API key is a meaningful first step, but it does not yet provide:

- user identity
- authorization
- richer abuse protection
- customer-facing output trust policy

So the security story has improved, but it is not complete.

---

## What Is Good Enough Today

Weekend Wizard is currently strong enough for:

- architecture review
- L2-level deliverable discussion
- controlled local demo
- operational learning around startup, evaluation, security, and observability

It now demonstrates:

- a clean planner/executor/reflection design
- MCP-backed tool execution
- grounded answers from real tool observations
- local startup reproducibility
- repeatable contract evaluation
- a customer-facing-style shared access boundary
- request-correlated phase telemetry

---

## What Still Blocks Trusted Production Use

Weekend Wizard is not yet production-ready in the stronger L3 sense.

Main blockers:

- latency is too high for a good interactive user experience
- heavier supported prompts can still time out
- the app remains highly sensitive to model choice and machine capability
- `staging` and `production` observability modes do not yet have a fully differentiated policy split
- evaluation currently measures contract behavior, not full answer quality or safety
- access control exists, but deeper customer-facing security and reliability hardening is still pending

---

## Recommended Next Steps

If this repo is used as an L3 learning branch, the next best steps are:

1. Complete the `staging` vs `production` observability split

Focus on:

- richer engineer-friendly staging logs
- stricter production event policy
- lower-noise production behavior

2. Use the telemetry evidence to guide latency decisions

This should drive decisions around:

- planner cost
- reflection cost
- timeout strategy
- prompt size and behavior tradeoffs

3. Expand evaluation maturity

This should become the basis for:

- broader prompt coverage
- unsupported-case handling
- regression detection

4. Continue customer-facing hardening deliberately

Likely next areas:

- deeper security hygiene
- reliability and degraded-dependency behavior
- operational documentation

---

## Bottom Line

Weekend Wizard now has a meaningfully stronger production-readiness foundation than it did on the main branch and than it did earlier in this `prod-readiness` branch.

The most important current conclusion is:

- architecture quality is no longer the main problem
- startup, evaluation, access control, and observability are all stronger
- request-phase evidence now exists
- local CPU-bound model latency is the main limiting factor

This is a useful, credible, and much more current result for the branch.
