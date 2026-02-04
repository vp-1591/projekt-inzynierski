import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template, train_on_responses_only

import requests
from transformers import TrainerCallback

class ProgressCallback(TrainerCallback):
    def __init__(self, backend_url):
        self.backend_url = backend_url

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.max_steps > 0:
            progress = int((state.global_step / state.max_steps) * 100)
            try:
                # print(f"DEBUG: Reporting progress {progress}% to {self.backend_url}", file=sys.stderr, flush=True)
                requests.post(f"{self.backend_url}/training/progress", 
                              json={"stage": "training", "value": progress},
                              timeout=1)
            except Exception as e:
                print(f"Failed to report progress to {self.backend_url}: {e}", file=sys.stderr, flush=True)

class ModelTrainer:
    def __init__(self, base_model="unsloth/bielik-7b-v1.1-bnb-4bit", output_dir="./model/latest"):
        self.base_model = base_model
        self.output_dir = output_dir
        self.max_seq_length = 2048

    def run_sft(self, dataset_path, max_steps=60, backend_url="http://localhost:8000"):
        """
        Implementation of 2.2: SFT with QLoRA & Long Context
        NOTE: This requires a Linux environment (or WSL2) and a compatible GPU.
        """
        if os.name == 'nt':
            print("WARNING: Unsloth is optimized for Linux. Running on Windows may fail.")

        import sys
        import subprocess
        # DEBUG: Print REAL system VRAM via nvidia-smi
        try:
            cmd = "nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv,noheader"
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            total, used, free = output.split(',')
            print(f"DEBUG [nvidia-smi]: Total: {total}, Used: {used}, Free: {free}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"DEBUG: Could not run nvidia-smi: {e}", file=sys.stderr, flush=True)

        max_seq_length = 1024 # Increased to fit 795t data, safe with Paged AdamW
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.base_model,
            max_seq_length = max_seq_length,
            load_in_4bit = True,
            use_gradient_checkpointing = "unsloth",
        )

        # 2. Add LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0,
            bias = "none",
            use_gradient_checkpointing = "unsloth",
            random_state = 3407,
            use_rslora = False,
            loftq_config = None,
        )

        # 3. Load Data
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        
        # 4. Setup Trainer
        from transformers import TrainingArguments
        
        def formatting_prompts_func(examples):
            convos = []
            texts = []
            mapper = {"input": "user", "output": "assistant"}
            for input_text, output_text in zip(examples["input"], examples["output"]):
                # Manual Truncation: 1024 tokens ~= 4000 chars. 
                # We limit input article to 3500 chars to save room for output and system prompts.
                # This prevents OOM on 16k token articles.
                truncated_input = input_text[:3500] + "...(truncated)" if len(input_text) > 3500 else input_text
                
                messages = [
                    {"role": "user", "content": truncated_input},
                    {"role": "assistant", "content": output_text}
                ]
                texts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False))
            return texts

        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False,
            formatting_func = formatting_prompts_func,
            args = TrainingArguments(
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 4, # Reduced from 8 to save interaction memory
                warmup_steps = 5,
                gradient_checkpointing = True, # CRITICAL FIX for VRAM
                num_train_epochs = 2,
                learning_rate = 2e-4,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "paged_adamw_8bit", # CRITICAL: Offload optimizer to RAM
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = self.output_dir,
            ),
            callbacks=[ProgressCallback(backend_url)]
        )

        # 5. Execute Training
        trainer_stats = trainer.train()
        
        # 6. Save Adapter (HF)
        adapter_path = f"{self.output_dir}/adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        # 7. Save Adapter (GGUF) - MOVED TO DEPLOYMENT PHASE for speed
        # GGUF conversion removed from training loop to save time during experimentation.

        # Return absolute path to avoid ambiguity (HF path)
        return os.path.abspath(adapter_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to training .jsonl")
    parser.add_argument("--output", type=str, default="./model/latest", help="Output directory")
    parser.add_argument("--base", type=str, default="unsloth/mistral-7b-bnb-4bit", help="Base model path")
    parser.add_argument("--backend", type=str, default="http://localhost:8000", help="Backend URL")
    args = parser.parse_args()

    # Note: Inside WSL, make sure path exists
    trainer_instance = ModelTrainer(base_model=args.base, output_dir=args.output)
    print(f"Starting training on {args.data}...")
    adapter_path = trainer_instance.run_sft(dataset_path=args.data, backend_url=args.backend)
    print(f"Training finished. Adapter saved to: {adapter_path}")

    # Notify backend (from inside WSL to Windows Host)
    import requests
    try:
        requests.post(f"{args.backend}/training/complete?adapter_path={adapter_path}")
        print("Backend notified of completion.")
    except Exception as e:
        print(f"Failed to notify backend: {e}")
