# Investigation Report: PicklingError During Training Checkpoint Save

## Summary

Training crashes with `_pickle.PicklingError: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>: it's not the same object as trl.trainer.sft_config.SFTConfig` during checkpoint saving. The root cause is a class identity mismatch created by Unsloth's monkey-patching of `trl.SFTConfig` combined with a stale reference in the JIT-compiled `UnslothSFTTrainer` module.

## Root Cause

Three classes coexist at runtime, and a stale reference in the compiled module causes `self.args` to be an instance of the **original** `trl.trainer.sft_config.SFTConfig` — while pickle resolves the same qualified path to the **patched** `UnslothSFTConfig`:

| Class | `__module__` | `__name__` | id | Role |
|---|---|---|---|---|
| Original `SFTConfig` | `trl.trainer.sft_config` | `SFTConfig` | A | Parent of UnslothSFTConfig; **used by compiled module's `SFTConfig` import** |
| Unsloth's `UnslothSFTConfig` (from `unsloth/trainer.py`) | `UnslothSFTTrainer` | `UnslothSFTConfig` | B | Created by unsloth; **replaces** `trl.trainer.sft_config.SFTConfig` at module level |
| Compiled module's `UnslothSFTConfig` (line 363) | `UnslothSFTTrainer` | `UnslothSFTConfig` | C | Subclass of original `SFTConfig`; used when `args is None` |

**The chain of events:**

1. **Unsloth patches** `trl.trainer.sft_config.SFTConfig` → `UnslothSFTConfig` (id B) and `trl.trainer.sft_trainer.SFTConfig` → same `UnslothSFTConfig` (id B).

2. **JIT-compiled module** (`UnslothSFTTrainer.py`) is generated. Line 31 does `from trl.trainer.sft_trainer import SFTConfig`. At import time, `trl.trainer.sft_trainer.SFTConfig` has **already been patched** to `UnslothSFTConfig` (id B). However, Python's `from X import Y` creates a name binding — it captures the object at import time.

3. **But the compiled module's `SFTConfig` is the ORIGINAL `trl.trainer.sft_config.SFTConfig` (id A), NOT the patched version (id B).** This is because Unsloth's `UnslothSFTConfig` (id B) was defined as `class UnslothSFTConfig(SFTConfig):` in the unsloth source, where `SFTConfig` was the **original** class. When unsloth replaces `trl.trainer.sft_config.SFTConfig` with `UnslothSFTConfig` (id B), the module-level name `SFTConfig` in `trl.trainer.sft_config` now points to id B — but the compiled module's import captured the original class (id A) because the import resolution followed: `trl.trainer.sft_trainer.SFTConfig` → which after patching is id B, but the compiled module seems to have gotten id A. *(Note: the exact mechanism may involve import ordering or the compiled module generator capturing the original class reference before the final patch.)*

4. In `_UnslothSFTTrainer.__init__` (line 791-795), when `args` is a `TrainingArguments` instance (as in our `trainer.py`), it converts it: `args = SFTConfig(**dict_args)`. This uses the compiled module's `SFTConfig` (id A) — creating an instance of the **original** `trl.trainer.sft_config.SFTConfig`.

5. At checkpoint save time, `torch.save(self.args, ...)` pickles `self.args`. Pickle sees `type(self.args).__module__ = 'trl.trainer.sft_config'` and `type(self.args).__name__ = 'SFTConfig'`, then looks up `sys.modules['trl.trainer.sft_config'].SFTConfig` — which now resolves to `UnslothSFTConfig` (id B). Since id A ≠ id B, pickle throws `PicklingError`.

## Evidence

### Training log (training_2.log, final lines)

```
_pickle.PicklingError: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>: it's not the same object as trl.trainer.sft_config.SFTConfig
```

Full traceback:
```
trainer.py:129  → trainer.train()
unsloth/trainer.py:844  → _train_with_reset
UnslothSFTTrainer.py:101  → wrapper
transformers/trainer.py:1424  → train → inner_training_loop
transformers/trainer.py:1775  → _run_epoch → _maybe_log_save_evaluate
transformers/trainer.py:2088  → _save_checkpoint
UnslothSFTTrainer.py:1380  → _save_checkpoint → super()._save_checkpoint
transformers/trainer.py:3046  → _save_checkpoint → save_model → _save
transformers/trainer.py:3840  → torch.save(self.args, ...)
```

### Runtime class identity test (inside container)

```
compiled.SFTConfig:           <class 'trl.trainer.sft_config.SFTConfig'>  (id A)
trl.trainer.sft_config.SFTConfig: <class 'UnslothSFTTrainer.UnslothSFTConfig'>  (id B)
Are they the same object? False

Instance.__class__.__module__: trl.trainer.sft_config
Instance.__class__.__name__:   SFTConfig
Resolved via sys.modules:      UnslothSFTConfig (id B)
Are they the same object? False

pickle.dumps FAILED: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>: it's not the same object as trl.trainer.sft_config.SFTConfig
torch.save FAILED: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>: it's not the same object as trl.trainer.sft_config.SFTConfig
```

### Why evaluation never starts

The crash occurs during `trainer.train()` at checkpoint save. The trainer process exits with a non-zero return code. Since the crash happens before `model.save_pretrained()` (line 132-134) and before `POST /training/complete` (line 152-158), the orchestrator never receives the completion notification, and `finish_training_and_evaluate()` is never called.

## Ruled Out

| Hypothesis | Why eliminated |
|---|---|
| OOM / GPU memory error | Training log shows 100% completion of all 16 steps; GPU memory is stable |
| Ollama connectivity issue | Error occurs in Python's pickle module, not network |
| Bug in `trainer.py` application code | The `TrainingArguments` → `SFTConfig` conversion happens inside Unsloth's compiled `_UnslothSFTTrainer.__init__`, not in `trainer.py` |
| Using `SFTConfig` instead of `TrainingArguments` would avoid the conversion entirely | Verified — but the bug is in Unsloth's compiled module, not in our code |
| `UnslothSFTConfig` (compiled module's line 363) is the type being pickled | Verified it's NOT — the actual type is the **original** `SFTConfig` (id A), because the compiled module's `SFTConfig` import captured the original class |
| `torch.save` bug | `pickle.dumps()` also fails with the same error — it's a Python pickle issue |

## Recommended Fix

**Option A (Minimal — use `SFTConfig` instead of `TrainingArguments`):**

In `backend/app/training/trainer.py`, replace `TrainingArguments(...)` with `SFTConfig(...)`. This avoids the `TrainingArguments` → `SFTConfig` conversion path entirely, since `SFTConfig` is already a subclass of `TrainingArguments`, so `isinstance(args, SFTConfig)` will be `True` and the problematic conversion branch at line 791-795 will be skipped.

```python
# Change line 9 from:
from trl import SFTTrainer, SFTConfig

# Change lines 109-124 from:
args = TrainingArguments(
    per_device_train_batch_size=1,
    ...
)

# To:
args = SFTConfig(
    per_device_train_batch_size=1,
    ...
)
```

Note: Some `TrainingArguments` parameters may not exist in `SFTConfig`. Verify parameter compatibility (e.g., `push_to_hub_token` was already being removed in the conversion). The `output_dir` parameter is required by both.

**Option B (Patching the compiled module — fragile, not recommended):**

Add a post-import fix in `trainer.py` that replaces the stale `SFTConfig` reference in the compiled module:

```python
import sys
if 'UnslothSFTTrainer' in sys.modules:
    compiled_mod = sys.modules['UnslothSFTTrainer']
    from trl import SFTConfig as patched_sftconfig
    compiled_mod.SFTConfig = patched_sftconfig
```

This is fragile because it depends on Unsloth internals and the compiled module being loaded before the fix runs.

**Option C (Pin unsloth version — wait for upstream fix):**

This is an Unsloth bug. The compiled module should not capture a stale reference to the original `SFTConfig`. A proper fix would be for Unsloth to ensure the compiled module's `SFTConfig` name binding points to the patched class. Reporting this to Unsloth is recommended, but Option A provides an immediate workaround.

**Recommendation:** Go with **Option A** — it's a one-line change (plus import adjustment) that avoids the class identity mismatch entirely, with no dependency on Unsloth internals.