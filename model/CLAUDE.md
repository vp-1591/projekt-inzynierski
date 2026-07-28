# model/CLAUDE.md

## Ollama API Compatibility

Ollama v0.5.5+ (January 2025) removed Modelfile-based input from `POST /api/create`.
The old format `{"name": "...", "modelfile": "..."}` returns:
`{"error":"neither 'from' or 'files' was specified"}`

The new JSON-based format requires:
1. Upload files via `POST /api/blobs/{sha256_digest}` (binary body = file content)
2. Call `POST /api/create` with:
   - `name`: model name
   - `files`: `{"filename.gguf": "sha256:..."}` for base model
   - `adapters`: `{"adapter.gguf": "sha256:..."}` for LoRA
   - `template`, `system`, `parameters`: configuration (same fields as Modelfile)

The `ollama create` CLI command still accepts Modelfiles — it converts internally.
The `ollama-entrypoint.sh` uses the CLI and is unaffected.

## Files in this directory

- `Modelfile` — Local dev Modelfile (absolute Windows paths). Used by `ollama create` CLI locally.
- `Modelfile.docker` — Docker Modelfile (paths use `/model/` mount point). Used by `ollama-entrypoint.sh` on container startup.
- `ollama-entrypoint.sh` — Custom entrypoint that creates the model via CLI on first boot.
- `bielik-4.5b-base/` — Base GGUF model.
- `xai-adapter/` — Production LoRA adapter checkpoints.
- `dataset/` — Training and test datasets.
- `benchmark-reports/` — Evaluation reports.