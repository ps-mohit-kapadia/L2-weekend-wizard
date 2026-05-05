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
