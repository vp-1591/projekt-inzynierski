import os

os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys

import requests
import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


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

        def formatting_prompts_func(example):
            """Prepares a single conversation for the chat template with truncation."""
            truncated_input = example["input"][:3500] if len(example["input"]) > 3500 else example["input"]
            messages = [
                {"role": "user", "content": truncated_input},
                {"role": "assistant", "content": example["output"]},
            ]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

        # Configure SFT Trainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            formatting_func=formatting_prompts_func,
            args=SFTConfig(
                dataset_text_field="text",
                max_length=max_seq_length,
                dataset_num_proc=2,
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
            ),
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
