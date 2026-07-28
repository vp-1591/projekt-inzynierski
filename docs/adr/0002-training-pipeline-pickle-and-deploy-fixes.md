# ADR 0002: Training pipeline pickle and deployment fixes

## Context

The MLOps training pipeline (train → evaluate → convert → deploy) crashed at multiple points when running with Unsloth SFT in the Docker container:

1. **Pickle crash**: `SFTTrainer.__init__()` calls `Dataset.map()` with `num_proc > 1` (auto-computed by Unsloth's compiled trainer), which uses `dill` for multiprocess serialization. The Unsloth tokenizer contains `ConfigModuleInstance` objects that `dill` cannot pickle, causing `TypeError: cannot pickle 'ConfigModuleInstance' object`.

2. **Empty logits crash**: Unsloth's memory optimization returns sentinel `EMPTY_LOGITS` instead of real logits. TRL's `compute_loss` calls `entropy_from_logits(outputs.logits)`, which fails with `TypeError: 'function' object is not subscriptable` because the sentinel's `__getattr__` returns function references instead of raising.

3. **Triton C compiler missing**: Unsloth uses Triton for GPU kernel JIT compilation, which requires `gcc`/`g++`. The Docker runtime stage only had Python packages — no C compiler.

4. **Deployment path mismatch and timeout**: The orchestrator's `deploy_new_adapter()` used `subprocess.run(docker exec ollama create ...)` which (a) wrote backend-container paths (`/app/model/...`) into `Modelfile.docker` that the Ollama container couldn't resolve, and (b) timed out at 120s because Ollama quantization (~96s) plus docker-exec overhead exceeded the limit.

5. **GGUF extension missing**: `convert_lora_to_gguf.py` was called with `--output {adapter_path}_gguf` (no `.gguf` extension). Ollama's `ADAPTER` directive requires the file path to end in `.gguf`.

## Decision

1. **Monkey-patch `Dataset.map`** at module level in `trainer.py` to force `num_proc=None` (single-process). This is the only reliable fix because Unsloth's compiled trainer auto-computes and overrides `dataset_num_proc` in `SFTConfig`, making the explicit setting ineffective. A defensive comment was also added in `benchmark.py`.

2. **Set `UNSLOTH_RETURN_LOGITS=1`** environment variable before any Unsloth imports. This forces Unsloth to return real logits tensors instead of the sentinel, preventing the `TypeError` in TRL's loss computation.

3. **Add `gcc g++` to the Docker runtime stage** alongside the Docker CLI installation. These are needed at runtime for Triton's JIT GPU kernel compilation.

4. **Replace `subprocess.run(docker exec)` with Ollama HTTP API**: `deploy_new_adapter()` now calls Ollama `/api/create` via `httpx.AsyncClient.stream()` with a 300s timeout. Modelfile content is rewritten in-memory (`/app/model/` → `/model/`) and sent in the JSON request body — no `Modelfile.docker` written to disk. Streaming NDJSON responses are parsed for error detection. This eliminates docker-exec overhead, avoids the 120s subprocess timeout, and keeps the asyncio event loop responsive.

5. **Change converter output suffix** from `{adapter_path}_gguf` to `{adapter_path}.gguf`. Add a three-tier GGUF file search fallback: primary `.gguf` path → legacy `_gguf` suffix → directory scan for any `.gguf` file.

Alternatives considered:
- **Set `dataset_num_proc=1` in SFTConfig** — doesn't work because Unsloth's compiled trainer overrides it.
- **Use `dill` with a custom pickler** — overly complex; the `ConfigModuleInstance` objects are fundamentally unpicklable.
- **Merge LoRA into base model before GGUF conversion** — possible but much slower; the ADAPTER directive is the intended Ollama deployment path for LoRA adapters.

## Constraints

- The `Dataset.map` monkey-patch must remain at module level (before any imports) because Unsloth patches it during import.
- `UNSLOTH_RETURN_LOGITS=1` must be set before Unsloth imports (it reads the env var at import time).
- The Ollama container and backend container share the `./model` volume but at different mount points (`/model/` vs `/app/model/`). Modelfile paths are rewritten in-memory to use the Ollama mount point before sending to the API.
- The HTTP deployment uses `httpx.AsyncClient.stream()` with a 300s timeout. The streaming response is parsed line-by-line for error detection (`{"error": "..."}` keys in NDJSON).
- `OLLAMA_API_URL` env var (default `http://ollama:11434`) is used for the HTTP endpoint. The `OLLAMA_CONTAINER` env var and `subprocess`-based docker-exec approach have been removed.
- The GGUF adapter file must have a `.gguf` extension for Ollama's `ADAPTER` directive.

## Consequences

- **Positive**: Training pipeline runs end-to-end without crashes: train → evaluate → convert → deploy → hot-swap.
- **Positive**: Deployment correctly maps paths between containers and creates the `bielik-lora-mipd` model in Ollama.
- **Positive**: HTTP-based deployment eliminates docker-exec overhead and subprocess timeout. The 300s httpx timeout provides ample headroom for quantization. No `Modelfile.docker` written to disk — Modelfile content is sent in the request body.
- **Positive**: The asyncio event loop is no longer blocked by synchronous `subprocess.run` calls during deployment.
- **Negative**: Single-process dataset tokenization is slower than multiprocess, but datasets are small enough (≤1521 documents) that this is negligible.
- **Negative**: The monkey-patch is fragile — if `Dataset.map`'s signature changes, the patch needs updating.

## Validation

1. Training completes (16 steps, loss decreasing) without pickle or logits errors.
2. Evaluation runs successfully and produces F1/exact-match metrics.
3. `POST /api/create` to Ollama HTTP API succeeds and creates the `bielik-lora-mipd` model.
4. Inference with the deployed model returns valid JSON with `reasoning` and `discovered_techniques` fields.
5. All 99 backend unit tests pass.