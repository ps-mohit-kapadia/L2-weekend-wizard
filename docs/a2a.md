# Weekend Wizard A2A Adapter

Weekend Wizard exposes a minimal synchronous A2A-compatible adapter for agent-to-agent integration.

This adapter is an access boundary only. It does not create a second agent brain, bypass the orchestrator, or call tools directly.

## Supported A2A Subset

- Agent Card discovery at `GET /.well-known/agent.json`
- JSON-RPC 2.0 execution at `POST /a2a/jsonrpc`
- `message/send`
- text input
- text artifact output
- synchronous completed response

## Unsupported A2A Features

- streaming responses
- task polling
- cancellation
- push notifications
- file, image, or binary input
- signed Agent Cards
- distributed task persistence
- multi-turn A2A sessions

## Agent Card

The Agent Card advertises Weekend Wizard capabilities, skills, endpoint, protocol version, and supported input/output modes.

**Protocol Version:** Weekend Wizard uses A2A protocol version `0.3.0`. This version was selected as the latest stable A2A specification at the time of implementation, ensuring compatibility with current A2A client implementations while providing access to the core `message/send` functionality.

## Request Example

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [
        {
          "kind": "text",
          "text": "Give me one trivia question."
        }
      ]
    }
  }
}
```

## Response Example

```json
{
  "jsonrpc": "2.0",
  "id": "req-1",
  "result": {
    "status": {
      "state": "completed"
    },
    "artifacts": [
      {
        "name": "weekend_wizard_answer",
        "parts": [
          {
            "kind": "text",
            "text": "Trivia: ..."
          }
        ]
      }
    ]
  }
}
```

## Security and Runtime Controls

The A2A adapter reuses the same API boundary controls as `/chat`:

- optional `X-API-Key` when `WEEKEND_WIZARD_API_KEY` is configured
- prompt size limit
- in-memory local/demo rate limit
- request timeout
- readiness checks

## Production Trust Notes

The adapter is intentionally minimal and honest. It proves that Weekend Wizard can be discovered and invoked by another agent through a documented protocol boundary while preserving the same hardened runtime lifecycle used by the HTTP API and CLI.
