# Agent Runs

This file records completed implementation runs only.

It should not be updated for planning, classification, read-only review, or read-only audit tasks.

## 2026-05-05 - Remove repeated observation parsing
MODE_USED: PERFORMANCE FIX
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: agent/orchestrator.py, agent/grounding.py
VERIFICATION_RUN: .\.venv\Scripts\python -m unittest tests.integration.test_orchestrator tests.unit.test_prompting
REGRESSION_RISK: low
INSTRUCTIONS_IGNORED: none
OUTCOME: Tool payloads are parsed once during execution and reused during finalization and coordinate updates.
FOLLOW_UP_NEEDED: none

## 2026-05-05 - Harden finalization against stale parsed payload cache
MODE_USED: SURGICAL FIX
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: agent/orchestrator.py, tests/integration/test_orchestrator.py
VERIFICATION_RUN: python -m pytest tests\integration\test_orchestrator.py; python -m unittest tests.integration.test_orchestrator
REGRESSION_RISK: medium
INSTRUCTIONS_IGNORED: none
OUTCOME: Finalization now treats tool observations as canonical and only reuses parsed payload cache when it is complete and consistent with observations.
FOLLOW_UP_NEEDED: Install missing test dependencies if this environment should execute orchestrator tests locally.

## 2026-05-05 - Preserve grounded degraded answers on required tool failure
MODE_USED: SURGICAL FIX
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: agent/orchestrator.py, tests/integration/test_orchestrator.py
VERIFICATION_RUN: python -m unittest tests.integration.test_orchestrator
REGRESSION_RISK: medium
INSTRUCTIONS_IGNORED: none
OUTCOME: Finalization now returns the grounded degraded draft when a required tool fails, preserving both unavailable required-tool content and successful optional tool output.
FOLLOW_UP_NEEDED: Install the missing local mcp dependency if orchestrator tests are expected to run in this environment.

## 2026-05-05 - Decouple external tool HTTP timeout from model timeout
MODE_USED: PERFORMANCE FIX
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: config/config.py, tools/shared.py, tests/unit/test_config.py, tests/unit/test_tools.py, .env.example, README.md
VERIFICATION_RUN: .\.venv\Scripts\python.exe -m unittest tests.unit.test_config tests.unit.test_tools
REGRESSION_RISK: low
INSTRUCTIONS_IGNORED: none
OUTCOME: External tool HTTP calls now use a dedicated bounded timeout setting without changing the longer Ollama request timeout used for planner and reflection calls.
FOLLOW_UP_NEEDED: Re-run the evaluation suite after restarting the API so the new environment setting is loaded.

## 2026-05-05 - Add dog-only planner fallback regression test
MODE_USED: SURGICAL FIX
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: tests/integration/test_orchestrator.py
VERIFICATION_RUN: .\.venv\Scripts\python.exe -m unittest tests.integration.test_orchestrator
REGRESSION_RISK: medium
INSTRUCTIONS_IGNORED: none
OUTCOME: Added a focused dog-only orchestration regression test that currently fails, confirming the bug before production behavior is changed.
FOLLOW_UP_NEEDED: Fix requested-tool inference so dog photo prompts do not trigger weather semantics through substring matching.

## 2026-05-05 - Fix dog-only weather keyword collision
MODE_USED: SURGICAL FIX
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: guardrails/guardrails.py, tests/unit/test_policy.py
VERIFICATION_RUN: .\.venv\Scripts\python.exe -m unittest tests.unit.test_policy tests.integration.test_orchestrator
REGRESSION_RISK: low
INSTRUCTIONS_IGNORED: none
OUTCOME: Weather inference now uses token matching, so dog photo prompts no longer accidentally request weather through the word "photo".
FOLLOW_UP_NEEDED: Re-run the evaluation suite to confirm the live dog-only case is now green.

## 2026-05-05 - Remove unused deterministic analyze_request path
MODE_USED: REFACTOR
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: guardrails/guardrails.py, guardrails/__init__.py, tests/unit/test_policy.py
VERIFICATION_RUN: .\.venv\Scripts\python.exe -m unittest tests.unit.test_policy tests.integration.test_orchestrator
REGRESSION_RISK: low
INSTRUCTIONS_IGNORED: none
OUTCOME: Removed the unused RequestAnalysis/analyze_request helper path and its dead supporting book-topic/book-limit logic without affecting the live planner/executor flow.
FOLLOW_UP_NEEDED: None.

## 2026-05-05 - Classify evaluation timeout-budget failures separately
MODE_USED: REFACTOR
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: evaluations/run_evaluations.py, tests/unit/test_evaluations.py
VERIFICATION_RUN: .\.venv\Scripts\python.exe -m unittest tests.unit.test_evaluations
REGRESSION_RISK: low
INSTRUCTIONS_IGNORED: none
OUTCOME: Evaluation summaries now separate timeout-budget failures from other contract failures without changing pass/fail behavior or exit status.
FOLLOW_UP_NEEDED: None.

## 2026-05-05 - Add books-only planner minimality regression
MODE_USED: SURGICAL FIX
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: tests/integration/test_orchestrator.py
VERIFICATION_RUN: .\.venv\Scripts\python.exe -m unittest tests.integration.test_orchestrator
REGRESSION_RISK: low
INSTRUCTIONS_IGNORED: none
OUTCOME: Added a focused books-only orchestration regression proving the planner/runtime path executes only book_recs for an unambiguous books-only prompt.
FOLLOW_UP_NEEDED: None.

## 2026-05-05 - Align docs with current planner/degrade/timeout behavior
MODE_USED: REFACTOR
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: README.md, documents/operator_runbook.md, .env.example
VERIFICATION_RUN: none
REGRESSION_RISK: low
INSTRUCTIONS_IGNORED: none
OUTCOME: Updated README, runbook, and env comments to reflect current planner/decide/reflect flow, degraded fallback behavior, local Ollama latency expectations, eval timeout-budget categories, and tool timeout guidance.
FOLLOW_UP_NEEDED: None.

## 2026-05-05 - Sanitize API error responses
MODE_USED: SURGICAL FIX
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: api.py, tests/unit/test_api.py
VERIFICATION_RUN: .\.venv\Scripts\python.exe -m unittest tests.unit.test_api
REGRESSION_RISK: low
INSTRUCTIONS_IGNORED: none
OUTCOME: Unexpected /chat failures now return a stable generic 500 message, and startup/not-ready responses no longer expose raw internal exception strings while detailed server logs are preserved.
FOLLOW_UP_NEEDED: None.

## 2026-05-05 - Offload live /ready recomputation from async request path
MODE_USED: SURGICAL FIX
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: api.py, tests/unit/test_api.py
VERIFICATION_RUN: .\.venv\Scripts\python.exe -m unittest tests.unit.test_api
REGRESSION_RISK: low
INSTRUCTIONS_IGNORED: none
OUTCOME: The /ready endpoint now offloads live readiness recomputation to a worker thread, preserving startup behavior and response shape while avoiding blocking Ollama I/O on the async request path.
FOLLOW_UP_NEEDED: None.

## 2026-05-05 - Validate numeric config settings
MODE_USED: SURGICAL FIX
SUBAGENTS_USED: none
APPROVAL_WAITED: yes
FILES_CHANGED: config/config.py, tests/unit/test_config.py
VERIFICATION_RUN: .\.venv\Scripts\python.exe -m unittest tests.unit.test_config
REGRESSION_RISK: low
INSTRUCTIONS_IGNORED: none
OUTCOME: Numeric environment settings now fail fast with clear field-specific errors for invalid ports, non-positive timeouts, and negative retry values while keeping valid defaults and zero-allowed retry settings unchanged.
FOLLOW_UP_NEEDED: None.
