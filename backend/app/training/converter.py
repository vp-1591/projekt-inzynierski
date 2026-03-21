import os
import sys
import subprocess

def main():
    """CLI utility for converting HuggingFace adapters to GGUF format using llama.cpp."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter HF directory")
    parser.add_argument("--base", type=str, required=False, help="Path to base model directory")
    parser.add_argument("--base-model-id", type=str, required=False, help="Base model ID (e.g. speakleash/Bielik-4.5B-v3)")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF file path (including .gguf extension)")
    parser.add_argument("--quant_method", type=str, default="q4_k_m", help="Quantization method (q4_k_m, f16, etc.)")
    args = parser.parse_args()

    # Environment Validation
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not found in environment; gated model access may fail.")

    if not args.base and not args.base_model_id:
        print("ERROR: Either --base or --base-model-id must be provided")
        sys.exit(1)

    # Locate the vendored llama.cpp conversion script
    # Path Resolution Logic
    script_path = os.path.abspath("backend/vendor/llama.cpp/convert_lora_to_gguf.py")
    
    if not os.path.exists(script_path):
        current_file_dir = os.path.dirname(os.path.abspath(__file__)) 
        project_root = os.path.dirname(os.path.dirname(current_file_dir))
        script_path = os.path.join(project_root, "vendor", "llama.cpp", "convert_lora_to_gguf.py")

    if not os.path.exists(script_path):
        script_path = "backend/vendor/llama.cpp/convert_lora_to_gguf.py"
    
    if not os.path.exists(script_path):
        print(f"ERROR: Conversion script not found at {script_path}")
        sys.exit(1)

    # Construct and Execute Command
    cmd = [
        "python3",
        script_path,
        "--outfile", args.output,
        args.adapter 
    ]

    if args.base_model_id:
        cmd.extend(["--base-model-id", args.base_model_id])
    elif args.base:
        cmd.extend(["--base", args.base])
    
    try:
        subprocess.check_call(cmd)
        print(f"Conversion successful. GGUF saved to {args.output}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed (code {e.returncode})")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
