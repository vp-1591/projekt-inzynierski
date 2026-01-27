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
                requests.post(f"{self.backend_url}/training/progress", 
                              json={"stage": "training", "value": progress},
                              timeout=1)
            except Exception as e:
                print(f"Failed to report progress: {e}")

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

        # 1. Load Model
        max_seq_length = 2048 # Reduced for developer version
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.base_model,
            max_seq_length = max_seq_length,
            load_in_4bit = True,
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
        trainer = SFTTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = dataset,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 2,
            packing = False,
            args = TrainingArguments(
                per_device_train_batch_size = 2,
                gradient_accumulation_steps = 4,
                warmup_steps = 5,
                max_steps = max_steps,
                learning_rate = 2e-4,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "linear",
                seed = 3407,
                output_dir = self.output_dir,
            ),
            callbacks=[ProgressCallback(backend_url)]
        )

        # 5. Execute Training
        trainer_stats = trainer.train()
        
        # 6. Save Adapter
        adapter_path = f"{self.output_dir}/adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        # Notify backend
        try:
            requests.post(f"{backend_url}/training/complete", json={"adapter_path": adapter_path})
        except:
            print("Failed to notify backend of completion")
            
        return trainer_stats

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to training .jsonl")
    parser.add_argument("--output", type=str, default="./model/latest", help="Output directory")
    parser.add_argument("--base", type=str, default="drive/MyDrive/bielik-4.5b-base", help="Base model path")
    args = parser.parse_args()

    # Note: Inside WSL, make sure path exists
    trainer_instance = ModelTrainer(base_model_path=args.base, output_dir=args.output)
    print(f"Starting training on {args.data}...")
    adapter_path = trainer_instance.run_sft(train_data_path=args.data)
    print(f"Training finished. Adapter saved to: {adapter_path}")

    # Notify backend (from inside WSL to Windows Host)
    import requests
    try:
        requests.post(f"http://127.0.0.1:8000/training/complete?adapter_path={adapter_path}")
        print("Backend notified of completion.")
    except Exception as e:
        print(f"Failed to notify backend: {e}")
