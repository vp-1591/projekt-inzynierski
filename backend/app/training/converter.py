import os
import sys
import torch
from unsloth import FastLanguageModel

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter HF directory")
    parser.add_argument("--base", type=str, required=True, help="Path to base model")
    parser.add_argument("--output", type=str, required=True, help="Output output GGUF DIRECTORY")
    parser.add_argument("--quant_method", type=str, default="q4_k_m", help="Quantization method (q4_k_m, q8_0, etc.)")
    args = parser.parse_args()

    print(f"DEBUG: Starting GGUF conversion for {args.adapter}")

    # 1. Load Model + Adapter
    # Note: To merge and save GGUF, we generally load in 16bit, but Unsloth handles 4bit/16bit complexity for us.
    # The key is we need to load the SAME setup we trained with or compatible one.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.adapter, # Load adapter directly
        max_seq_length = 2048,
        load_in_4bit = True, # We load in 4bit to merge with 4bit base if that matches training
        # If we trained on 4bit base, we should load 4bit base here.
    )

    # 2. Save GGUF
    # This automatically merges LoRA into base model and quantizes
    # Since we want a fast deploy, we might use "q4_k_m" (recommended) or "q8_0" for quality.
    # User asked for "merged_4bit" optimization effectively.
    
    print(f"DEBUG: Saving GGUF to {args.output} with method {args.quant_method}...")
    
    # "quantization_method" argument in save_pretrained_gguf accepts strings like "f16", "q4_k_m"
    # To use the optimized "forced_merged_4bit" (saves only adapter? No, that's save_pretrained_merged).
    # save_pretrained_gguf handles everything.
    
    # IMPORTANT: Output must be a directory? Or file?
    # Unsloth save_pretrained_gguf takes a directory and saves .gguf inside it.
    
    os.makedirs(args.output, exist_ok=True)
    
    model.save_pretrained_gguf(
        args.output, 
        tokenizer, 
        quantization_method=args.quant_method
    )
    
    print(f"DEBUG: Conversion complete. GGUF saved in {args.output}")

if __name__ == "__main__":
    main()
