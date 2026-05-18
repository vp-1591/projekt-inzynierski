# Agent Verification Notes

Preferred backend verification order:

1. Run `pytest --run-integration` when the local stack is available.
   This exercises the fast unit tests plus opt-in checks for Ollama, WSL,
   live FastAPI/uvicorn, WSL-originated progress callbacks, and real
   websocket broadcasts.
2. Run `pytest` when Ollama, WSL, or live backend prerequisites are not
   available. This runs deterministic unit coverage only.

Integration tests are expected to skip with explicit reasons when local
services or WSL prerequisites are missing.

# Agent Commit Notes

Commit messages should include a bullet list with concrete details of the
changes included in the commit.

For every change, append an entry to `feat.log` using this format:

`[timestamp | change(fix, feat, etc) | reasoning]`
