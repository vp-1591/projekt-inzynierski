# ADR 0002: Training pipeline pickle and deployment fixes

## Context

The MLOps training pipeline (train → evaluate → convert → deploy) crashed at multiple points when running with Unsloth SFT in the Docker container:

1. **Pickle crash**: `SFTTrainer.__init__()` calls `Dataset.map()` with `num_proc > 1` (auto-computed by Unsloth's compiled trainer), which uses `dill` for multiprocess serialization. The Unsloth tokenizer contains `ConfigModuleInstance` objects that `dill` cannot pickle, causing `TypeError: cannot pickle 'ConfigModuleInstance' object`.

2. **Empty logits crash**: Unsloth's memory optimization returns sentinel `EMPTY_LOGITS` instead of real logits. TRL's `compute_loss` calls `entropy_from_logits(outputs.logits)`, which fails with `TypeError: 'function' object is not subscriptable` because the sentinel's `__getattr__` returns function references instead of raising.

3. **Triton C compiler missing**: Unsloth uses Triton for GPU kernel JIT compilation, which requires `gcc`/`g++`. The Docker runtime stage only had Python packages — no C compiler.

4. **Deployment path mismatch**: The orchestrator's `deploy_new_adapter()` writes backend-container paths (`/app/model/...`) into the Modelfile, but the Ollama container mounts `./model` at `/model/` (not `/app/model/`). This caused `Error: stat /app/model/latest/adapter_gguf: no such file or directory`.

5. **GGUF extension missing**: `convert_lora_to_gguf.py` was called with `--output {adapter_path}_gguf` (no `.gguf` extension). Ollama's `ADAPTER` directive requires the file path to end in `.gguf`.

## Decision

1. **Monkey-patch `Dataset.map`** at module level in `trainer.py` to force `num_proc=None` (single-process). This is the only reliable fix because Unsloth's compiled trainer auto-computes and overrides `dataset_num_proc` in `SFTConfig`, making the explicit setting ineffective. A defensive comment was also added in `benchmark.py`.

2. **Set `UNSLOTH_RETURN_LOGITS=1`** environment variable before any Unsloth imports. This forces Unsloth to return real logits tensors instead of the sentinel, preventing the `TypeError` in TRL's loss computation.

3. **Add `gcc g++` to the Docker runtime stage** alongside the Docker CLI installation. These are needed at runtime for Triton's JIT GPU kernel compilation.

4. **Rewrite Modelfile paths for Ollama container**: In `deploy_new_adapter()`, replace `/app/model/` with `/model/` in both `FROM` and `ADAPTER` directives before writing `Modelfile.docker`.

5. **Change converter output suffix** from `{adapter_path}_gguf` to `{adapter_path}.gguf`. Add a three-tier GGUF file search fallback: primary `.gguf` path → legacy `_gguf` suffix → directory scan for any `.gguf` file.

Alternatives considered:
- **Set `dataset_num_proc=1` in SFTConfig** — doesn't work because Unsloth's compiled trainer overrides it.
- **Use `dill` with a custom pickler** — overly complex; the `ConfigModuleInstance` objects are fundamentally unpicklable.
- **Merge LoRA into base model before GGUF conversion** — possible but much slower; the ADAPTER directive is the intended Ollama deployment path for LoRA adapters.

## Constraints

- The `Dataset.map` monkey-patch must remain at module level (before any imports) because Unsloth patches it during import.
- `UNSLOTH_RETURN_LOGITS=1` must be set before Unsloth imports (it reads the env var at import time).
- The Ollama container and backend container share the `./model` volume but at different mount points (`/model/` vs `/app/model/`). All Modelfile paths must use the Ollama mount point.
- The GGUF adapter file must have a `.gguf` extension for Ollama's `ADAPTER` directive.

## Consequences

- **Positive**: Training pipeline runs end-to-end without crashes: train → evaluate → convert → deploy → hot-swap.
- **Positive**: Deployment correctly maps paths between containers and creates the `bielik-lora-mipd` model in Ollama.
- **Negative**: Single-process dataset tokenization is slower than multiprocess, but datasets are small enough (≤1521 documents) that this is negligible.
- **Negative**: The monkey-patch is fragile — if `Dataset.map`'s signature changes, the patch needs updating.

## Validation

1. Training completes (16 steps, loss decreasing) without pickle or logits errors.
2. Evaluation runs successfully and produces F1/exact-match metrics.
3. `ollama create bielik-lora-mipd -f /model/Modelfile.docker` succeeds in the Ollama container.
4. Inference with the deployed model returns valid JSON with `reasoning` and `discovered_techniques` fields.
5. All 99 backend unit tests pass.