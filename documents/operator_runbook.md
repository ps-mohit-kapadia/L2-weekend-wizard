# Weekend Wizard Operator Runbook

This runbook is the operator-facing workflow for starting, updating, verifying, and triaging the Weekend Wizard API service.

## Configuration

Use [C:\Users\MohitKapadiya\Desktop\New folder\genai\L2_agents\weekend-wizard\.env.example](C:/Users/MohitKapadiya/Desktop/New%20folder/genai/L2_agents/weekend-wizard/.env.example) as the template for local or deployment-specific environment settings.

Minimum required values:

- `WEEKEND_WIZARD_API_KEY`
- `WEEKEND_WIZARD_API_URL`
- `WEEKEND_WIZARD_REQUEST_TIMEOUT`
- `OLLAMA_URL`

Common runtime overrides:

- `WEEKEND_WIZARD_API_HOST`
- `WEEKEND_WIZARD_API_PORT`
- `WEEKEND_WIZARD_PREFERRED_MODELS`
- `WEEKEND_WIZARD_OBSERVABILITY_MODE`
- `WEEKEND_WIZARD_LOG_LEVEL`

## Startup Workflow

1. Activate the virtual environment.
2. Confirm `.env` is present and has the intended values.
3. Run:

```powershell
python .\scripts\dev_up.py check
```

4. If preflight passes, start the API:

```powershell
python .\scripts\dev_up.py api
```

5. Verify readiness:

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:8000/ready -Headers @{ "X-API-Key" = "<your-key>" }
```

6. Optionally run a smoke prompt:

```powershell
.\.venv\Scripts\python.exe .\tests\smoke\smoke_test.py --prompt "Tell me a joke."
```

## Update Workflow

Use this checklist after pulling new code or changing configuration.

1. Pull the latest code.
2. Review `.env` for any new or changed settings.
3. Run:

```powershell
python .\scripts\dev_up.py check
```

4. Restart the API process.
5. Verify `/ready`.
6. Run the smoke test.
7. If the update includes behavior changes, run:

```powershell
.\.venv\Scripts\python.exe .\evaluations\run_evaluations.py
```

## Restart Rules

Restart the API process after:

- editing `.env`
- changing model-related config
- changing runtime host/port
- pulling new code that touches startup, orchestration, guardrails, or tools

## Failure Triage

### `/ready` returns `503`

Check, in order:

1. Is Ollama reachable?
2. Is the configured preferred model available?
3. Can the MCP runtime start?
4. Were any tools discovered successfully?

Best first command:

```powershell
python .\scripts\dev_up.py check
```

### `/chat` returns `401`

Check:

- `WEEKEND_WIZARD_API_KEY` on the server
- matching `X-API-Key` from the client, smoke test, or evaluation runner

### `/chat` returns `503`

Check:

- the API was restarted after config changes
- `/ready` is green
- Ollama/model/tool discovery is still healthy

### `/chat` returns `200` with `response_status=degraded`

Check:

- planner fallback
- required-tool failures in logs
- tool observations in the response payload

Degraded responses are valid transport-level responses, but they should be treated as product-quality issues during verification.

## Verification Signals

Healthy runtime signs:

- `/health` returns `200`
- `/ready` returns `200` with `status=ready`
- smoke test passes
- evaluations pass for the targeted prompt set
- logs show `outcome=success` for expected happy-path prompts

## Recovery Checklist

If a new change behaves badly:

1. Stop the API process.
2. Restore the last known-good code/config state.
3. Run `python .\scripts\dev_up.py check`.
4. Restart the API.
5. Verify `/ready`.
6. Re-run the smoke test.

## Notes

- `README.md` remains the project overview and local setup doc.
- This runbook is the operator-facing workflow doc.
