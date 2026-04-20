# Production Readiness Notes

## Purpose

This document captures the production-readiness learning work completed on the `prod-readiness` branch.

The goal of this branch was not to turn Weekend Wizard into a fully production-ready system, but to push the repo toward a more operationally mature shape and to document what currently blocks trusted deployment.

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

---

## What Is Good Enough Today

Weekend Wizard is currently strong enough for:

- architecture review
- L2-level deliverable discussion
- controlled local demo
- operational learning around startup and evaluation

It now demonstrates:

- a clean planner/executor/reflection design
- MCP-backed tool execution
- grounded answers from real tool observations
- local startup reproducibility
- repeatable contract evaluation

---

## What Still Blocks Trusted Production Use

Weekend Wizard is not yet production-ready in the stronger L3 sense.

Main blockers:

- latency is too high for a good interactive user experience
- heavier supported prompts can still time out
- the app remains highly sensitive to model choice and machine capability
- observability is still lightweight rather than operationally deep
- evaluation currently measures contract behavior, not full answer quality or safety

---

## Recommended Next Steps

If this repo is used as an L3 learning branch, the next best steps are:

1. Add request-phase timing visibility

Focus on:

- planning time
- tool execution time
- reflection time
- end-to-end request time

2. Use the evaluation harness for model and timeout comparison

This should become the basis for:

- model selection
- timeout policy
- regression detection

3. Keep optimization evidence-based

Do not optimize blindly.

Use measured timing to decide whether the next change should target:

- planner cost
- reflection cost
- timeout strategy
- prompt size

---

## Bottom Line

Weekend Wizard now has a stronger production-readiness foundation than it did on the main branch.

The most important conclusion from this branch is:

- architecture quality is no longer the main problem
- operational clarity is better
- evaluation evidence now exists
- local CPU-bound model latency is the main limiting factor

This is a useful and credible result.
