import os

os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # TRL's compute_loss calls entropy_from_logits(outputs.logits)
# which needs real logits tensors. Without this, Unsloth returns EMPTY_LOGITS (a
# sentinel whose __getattr__ returns function refs instead of raising), causing
# TypeError: 'function' object is not subscriptable on logits.shape[:-1].
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import functools
import sys

import requests
import torch
from datasets import Dataset, load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

# Monkey-patch Dataset.map to force num_proc=None (single-process).
# Unsloth's compiled SFTTrainer auto-computes dataset_num_proc (e.g. 4 on our
# container) and passes it to datasets.map(), which uses multiprocess/dill for
# serialization. The Unsloth tokenizer contains ConfigModuleInstance objects
# that dill cannot pickle, causing TypeError. Setting dataset_num_proc=1 in
# SFTConfig is NOT enough — Unsloth's __init__ overrides it. The only reliable
# fix is to intercept at the Dataset.map level. See Unsloth Issue #4490.
_original_dataset_map = Dataset.map


def _safe_map(self, *args, **kwargs):
    kwargs["num_proc"] = None
    return _original_dataset_map(self, *args, **kwargs)


Dataset.map = _safe_map


class ProgressCallback(TrainerCallback):
    """Callback to report training progress back to the backend API."""

    def __init__(self, backend_url):
        self.backend_url = backend_url

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.max_steps > 0:
            progress = int((state.global_step / state.max_steps) * 100)
            try:
                requests.post(
                    f"{self.backend_url}/training/progress", json={"stage": "training", "value": progress}, timeout=1
                )
            except Exception as e:
                print(f"Failed to report progress: {e}", file=sys.stderr, flush=True)


def format_training_example(example, tokenizer, max_input_length=3500):
    """Format a training example for SFT. Returns a plain string.

    TRL's SFTTrainer._prepare_dataset wraps the formatting function's return
    value in {"text": ...} automatically. Returning a dict would cause nested
    dicts that crash add_eos (AttributeError: 'dict' has no attribute 'endswith').
    """
    truncated_input = (
        example["input"][:max_input_length] if len(example["input"]) > max_input_length else example["input"]
    )
    messages = [
        {"role": "user", "content": truncated_input},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


class ModelTrainer:
    """Handles the Finetuning (SFT) process using Unsloth and QLoRA."""

    def __init__(self, base_model="unsloth/bielik-7b-v1.1-bnb-4bit", output_dir="./model/latest"):
        self.base_model = base_model
        self.output_dir = output_dir
        self.max_seq_length = 2048

    def run_sft(self, dataset_path, max_steps=60, backend_url="http://localhost:8000"):
        """
        Executes Supervised Fine-Tuning (SFT).
        Optimized for 4-bit quantization and gradient checkpointing to save VRAM.
        """
        import subprocess
        import sys

        # Check GPU memory status
        try:
            cmd = "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader"
            output = subprocess.check_output(cmd, shell=True, encoding="utf-8", errors="replace").strip()
            total, used, free = output.split(",")
            print(f"GPU Status: Total: {total}, Used: {used}, Free: {free}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"nvidia-smi check failed: {e}", file=sys.stderr, flush=True)

        max_seq_length = 1024  # Target sequence length for training

        # Load Model & Tokenizer with 4-bit quantization
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.base_model,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            local_files_only=True,
        )

        # Initialize LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        dataset = load_dataset("json", data_files=dataset_path, split="train")

        formatting_func = functools.partial(format_training_example, tokenizer=tokenizer)

        # dataset_num_proc is handled by the Dataset.map monkey-patch at module
        # level (see top of file). Setting it here in SFTConfig has no effect —
        # Unsloth's compiled trainer auto-computes and overrides it. See commit
        # c0af1e5 for the related TrainingArguments pickle fix.
        sft_config = SFTConfig(
            dataset_text_field="text",
            max_length=max_seq_length,
            packing=False,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            gradient_checkpointing=True,
            num_train_epochs=2,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=self.output_dir,
        )

        # Configure SFT Trainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            formatting_func=formatting_func,
            args=sft_config,
            callbacks=[ProgressCallback(backend_url)],
        )

        # Execute Training
        trainer.train()

        # Save Fine-tuned adapters
        adapter_path = f"{self.output_dir}/adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        return os.path.abspath(adapter_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to training .jsonl")
    parser.add_argument("--output", type=str, default="./model/latest", help="Output directory")
    parser.add_argument("--base", type=str, default="unsloth/mistral-7b-bnb-4bit", help="Base model path")
    parser.add_argument("--backend", type=str, default="http://localhost:8000", help="Backend URL")
    args = parser.parse_args()

    trainer_instance = ModelTrainer(base_model=args.base, output_dir=args.output)
    print(f"Starting training on {args.data}...")
    adapter_path = trainer_instance.run_sft(dataset_path=args.data, backend_url=args.backend)
    print(f"Training finished. Adapter saved to: {adapter_path}")

    # Notify backend of completion
    import requests

    try:
        requests.post(f"{args.backend}/training/complete?adapter_path={adapter_path}")
        print("Backend notified.")
    except Exception as e:
        print(f"Failed to notify backend: {e}")
