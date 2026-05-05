# Codex Delivery Agent Workflow

This repo uses a custom Codex delivery workflow to reduce repeated prompting, prevent plaster fixes, and make AI-assisted development safer for production work.

## Structure

```text
.codex/
  agent-runs.md
  README.md
  agents/
    delivery-agent.toml
    architecture-reviewer.toml
    root-cause-debugger.toml
    codebase-pattern-finder.toml
    test-coverage-reviewer.toml
    ux-state-reviewer.toml
    security-validation-reviewer.toml
    requirements-analyst.toml
    performance-reviewer.toml
    integration-contract-reviewer.toml
```

## Main Usage

Start normal Codex tasks with:

```text
Follow the repository delivery-agent instructions.

Task:
<describe the task>
```

For investigations:

```text
Follow the repository delivery-agent instructions.

Investigate:
<describe the concern>
```

For release or demo readiness:

```text
Follow the repository delivery-agent instructions.

Task:
Check if this app is demo-ready.
```

For workflow review:

```text
Follow the repository delivery-agent instructions.

Task:
Review .codex/agent-runs.md and suggest one workflow improvement.
```

## Modes

The delivery agent must classify every task into exactly one mode:

- `SURGICAL FIX` — small bug, regression, or failing test
- `PERFORMANCE FIX` — known repeated work or slow path to fix
- `FEATURE DELIVERY` — contained new product behavior
- `ARCHITECTURE / SYSTEM DESIGN` — boundaries, scaling, tenancy, integrations, or long-term structure
- `REFACTOR` — behavior-preserving structure cleanup
- `REVIEW` — production PR or diff review
- `PERFORMANCE AUDIT` — read-only performance bottleneck review
- `PROJECT DELIVERY` — multi-step end-to-end product/module work
- `INVESTIGATION` — read-only analysis to determine whether something is a real issue
- `RELEASE CHECK` — demo, release, or production-readiness review
- `AGENTOPS REVIEW` — review agent logs and workflow effectiveness

## Approval Rule

For non-trivial work, Codex must plan first and wait for approval before editing.

Approved phrases include:

```text
proceed
implement
apply
go ahead
approved
reclassify and proceed
```

Planning, review, audit, investigation, release check, and AgentOps review should not edit files.

## Subagents

Subagents are read-only helpers.

They may review, investigate, and plan.

They must not implement changes.

Only `delivery-agent` may implement changes.

Subagents are used only for complex, risky, unclear, architectural, stuck, or multi-layer work.

## AgentOps Logging

Implementation runs are logged in:

```text
.codex/agent-runs.md
```

The log should update only after:

- approved implementation
- completed trivial fix

The log should not update for:

- planning only
- classification only
- read-only review
- read-only audit
- investigation
- release check
- AgentOps review
- failed or aborted implementation
- unapproved non-trivial work

## Final Implementation Response Format

After approved implementation or completed trivial fix, Codex must respond with:

```text
DONE:
CHANGED:
VERIFIED:
RISKS:
AGENTOPS:
- MODE_USED:
- SUBAGENTS_USED:
- APPROVAL_WAITED:
- VERIFICATION_RUN:
- REGRESSION_RISK:
- INSTRUCTIONS_IGNORED:
- FOLLOW_UP_NEEDED:
```

## Regression Risk Rule

If verification cannot complete because of missing dependencies, environment issues, skipped tests, or unavailable services, `REGRESSION_RISK` should be `medium` unless another meaningful verification passed.

If a performance fix changes shared state, orchestration state, cache behavior, finalization, validation, or cross-layer data flow, downstream consumers must be verified before regression risk can be marked `low`.

## Common Prompts

### Normal Task

```text
Follow the repository delivery-agent instructions.

Task:
<task>
```

### Investigation

```text
Follow the repository delivery-agent instructions.

Investigate:
<concern>

Do not edit files.
```

### Performance Audit

```text
Follow the repository delivery-agent instructions.

Task:
Performance audit <scope> for bottlenecks, repeated work, time complexity issues, and scalability risks.

Do not edit files.
```

### Review Current Diff

```text
Follow the repository delivery-agent instructions.

Task:
Review the current diff like a production PR.

Do not edit files.
Focus on root-cause correctness, architecture, security/validation, test coverage, UX/state risks, and plaster fixes.
```

### Release Check

```text
Follow the repository delivery-agent instructions.

Task:
Check whether this app is demo-ready.

Do not edit files.
```

### AgentOps Review

```text
Follow the repository delivery-agent instructions.

Task:
Review .codex/agent-runs.md and tell me whether the delivery-agent workflow is helping or causing rework.

Do not edit files.
```

## How To Test The Workflow

Use fresh Codex chats when testing configuration changes.

### Routing Test

```text
Follow the repository delivery-agent instructions.

Do not edit files.

Classify these tasks only:

1. Fix a failing login button test.
2. Review agent/orchestrator.py for repeated parsing and time complexity issues.
3. Add invite-user flow with role selection, validation, API call, loading/error states, and tests.
4. Design a scalable multi-tenant settings architecture.
5. Review the current diff for production risks.
6. Refactor duplicated date-formatting helpers without behavior changes.
7. Fix repeated observation parsing in agent/orchestrator.py and agent/grounding.py.
8. Build a full billing module with plans, invoices, permissions, UI, API, and tests.

Return only:
TASK:
MODE:
SUBAGENTS NEEDED:
WHY:
```

Expected modes:

```text
1. SURGICAL FIX
2. PERFORMANCE AUDIT
3. FEATURE DELIVERY or PROJECT DELIVERY
4. ARCHITECTURE / SYSTEM DESIGN
5. REVIEW
6. REFACTOR
7. PERFORMANCE FIX
8. PROJECT DELIVERY
```

### Approval Behavior Test

```text
Follow the repository delivery-agent instructions.

Task:
Fix repeated observation parsing in:
- agent/orchestrator.py
- agent/grounding.py

Do not edit files yet.
Return only:
MODE:
WHY:
PLAN:
FILES:
```

Expected behavior:

```text
MODE: PERFORMANCE FIX
No edits
No AGENTOPS
No .codex/agent-runs.md update
Waits for approval
```

### Investigation Test

```text
Follow the repository delivery-agent instructions.

Investigate:
Repeated calls to the same tool may overwrite parsed payloads during finalization.

Do not edit files.
Return only:
MODE:
WHY:
PLAN:
FILES:
```

Expected behavior:

```text
MODE: INVESTIGATION
No edits
No AGENTOPS
No .codex/agent-runs.md update
No .codex files in FILES
```

### Release Check Test

```text
Follow the repository delivery-agent instructions.

Task:
Check whether this app is demo-ready.

Do not edit files.
Return only:
MODE:
WHY:
PLAN:
FILES:
```

Expected behavior:

```text
MODE: RELEASE CHECK
No edits
No AGENTOPS
No .codex/agent-runs.md update
```

### AgentOps Review Test

```text
Follow the repository delivery-agent instructions.

Task:
Review .codex/agent-runs.md and tell me whether the delivery-agent workflow is helping or causing rework.

Do not edit files.
Return only:
MODE:
WHY:
PLAN:
FILES:
```

Expected behavior:

```text
MODE: AGENTOPS REVIEW
No edits
No AGENTOPS
No .codex/agent-runs.md update
```

## When To Change This Workflow

Do not keep adding rules casually.

Only update the workflow when `.codex/agent-runs.md` or repeated real usage shows a repeated failure pattern.

Good reasons to update:

- wrong mode repeatedly selected
- agent edits without approval
- final response contract repeatedly missed
- verification skipped or misreported
- same issue needs repeated follow-up fixes
- subagents add noise or miss important risks

Bad reasons to update:

- one-off imperfect answer
- personal preference
- trying to make the agent perfect
- adding more subagents without evidence

## New Repo Setup

To reuse this workflow in a new repo, copy:

```text
AGENTS.md
.codex/
  README.md
  agent-runs.md
  agents/
    *.toml
```

Start the new repo with a fresh `.codex/agent-runs.md`.

Do not copy old run history into a new repo.