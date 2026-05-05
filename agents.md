# AGENTS.md

Follow the Engineering Delivery Agent workflow as the default behavior for this repository.

The detailed workflow lives in:

- `.codex/agents/delivery-agent.toml`
- `.codex/README.md`

The agent must classify each request before acting and use the exact mode names defined in `delivery-agent.toml`.

Repository rule:
Ship production-quality software with the smallest correct change for the selected mode.

Avoid:
- plaster fixes
- unnecessary architecture
- duplicate logic
- hardcoded business rules
- broad error masking
- unrelated cleanup
- large rewrites unless the selected mode requires them

Default behavior:
- For architecture or system design tasks, plan first.
- For small fixes, keep the diff small.
- For features, deliver complete product behavior.
- For performance, remove repeated work through ownership/data flow before considering caches.
- For investigations, audits, release checks, AgentOps reviews, and code reviews, inspect first and do not edit unless asked.
- For non-trivial implementation, wait for approval before editing.

Use `.codex/README.md` for how to operate, test, and maintain the Codex workflow.