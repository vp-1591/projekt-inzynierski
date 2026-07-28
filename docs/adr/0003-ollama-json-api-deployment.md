# ADR 0003: Ollama JSON-based `/api/create` deployment

## Status

active

Supersedes ADR 0002 (decision point 4 only — deployment mechanism)

## Context

ADR 0002 replaced `subprocess.run(docker exec)` with Ollama HTTP API, sending Modelfile content in the `POST /api/create` request body. This worked with Ollama versions before v0.5.5 (January 2025).

As of Ollama v0.5.5, the HTTP API no longer accepts Modelfile-based input. Sending `{"name": "...", "modelfile": "..."}` returns `{"error":"neither 'from' or 'files' was specified"}`. The `ollama create` CLI still accepts Modelfiles — it converts them internally — so the entrypoint script (`ollama-entrypoint.sh`) is unaffected.

The new API requires a two-phase approach:
1. Upload GGUF files as blobs via `POST /api/blobs/{sha256_digest}` (with `HEAD` check to skip already-uploaded blobs)
2. Call `POST /api/create` with JSON fields: `name`, `files`, `adapters`, `template`, `system`, `parameters`

The running Docker container uses `ollama/ollama:0.32.4`, which requires the new format.

## Decision

Replace the Modelfile-based HTTP deployment in `deploy_new_adapter()` with the new JSON-based API:

1. **Blob upload helper** (`_upload_blob()`): Computes SHA-256 digest, checks existence via `HEAD /api/blobs/{digest}` (skips if 200), then uploads via `POST /api/blobs/{digest}` with binary file content.

2. **JSON-based model creation**: `POST /api/create` with `files` (base GGUF filename → digest), `adapters` (adapter GGUF filename → digest), `template` (ChatML), `system` (system prompt), `parameters` (temperature, stop tokens, context length).

3. **Remove Modelfile reading and path rewriting**: The Modelfile is no longer needed for HTTP deployment. The `FROM` and `ADAPTER` lines, path rewriting (`/app/model/` → `/model/`), and regex ADAPTER replacement are all removed. Template, system prompt, and parameters are defined as constants in `orchestrator.py`.

4. **Remove Docker CLI and Docker socket mount**: Since deployment now uses HTTP API exclusively, the backend no longer needs Docker CLI installed or the Docker socket mounted. The Dockerfile was updated to remove Docker CLI installation, and `docker-compose.yml` no longer mounts `/var/run/docker.sock`.

5. **Pin Ollama version**: `docker-compose.yml` now uses `ollama/ollama:0.32.4` instead of `:latest` to prevent future breaking changes.

6. **Modelfiles preserved for CLI**: `model/Modelfile` (local dev) and `model/Modelfile.docker` (container entrypoint) are unchanged — the `ollama create` CLI still uses them.

The `ollama-entrypoint.sh` uses the CLI, not the HTTP API, so it is unaffected.

## Constraints

- Ollama v0.5.5+ is required for the JSON-based `/api/create` format.
- Blob upload sends raw file content from the backend container — the different mount points (`/app/model/` vs `/model/`) don't matter because blobs are uploaded by value, not by path reference.
- `files` and `adapters` dict keys are filenames only (e.g., `"Bielik-4.5B-v3.0-Instruct.Q8_0.gguf"`), not full paths.
- The ChatML template, system prompt, and model parameters are hardcoded as constants in `orchestrator.py` and must stay in sync with `model/Modelfile.docker`.

## Consequences

- **Positive**: Deployment works with current and future Ollama versions (v0.5.5+).
- **Positive**: Blob deduplication via `HEAD` check avoids re-uploading unchanged base model or adapter files.
- **Positive**: No Docker CLI dependency in the backend image — smaller image, smaller attack surface.
- **Positive**: No Docker socket mount needed — improved security posture.
- **Negative**: Template, system prompt, and parameters are duplicated as constants in `orchestrator.py` (must stay in sync with `model/Modelfile.docker`).
- **Negative**: Pinned Ollama version (`0.32.4`) must be updated manually when upgrading.

## Validation

1. All 106 backend unit tests pass (7 new deploy tests covering success, JSON payload, HTTP error, stream error, blob upload failure, no base GGUF, blob deduplication).
2. `POST /api/blobs/{digest}` upload and `HEAD` deduplication verified with live Ollama container (v0.32.4).
3. `POST /api/create` with JSON fields (`files`, `adapters`, `template`, `system`, `parameters`) creates model successfully.