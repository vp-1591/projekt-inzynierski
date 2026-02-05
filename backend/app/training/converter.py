import os
import sys
import subprocess

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to adapter HF directory")
    parser.add_argument("--base", type=str, required=False, help="Path to base model directory")
    parser.add_argument("--base-model-id", type=str, required=False, help="Base model ID (e.g. speakleash/Bielik-4.5B-v3)")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF file path (including .gguf extension)")
    parser.add_argument("--quant_method", type=str, default="q4_k_m", help="Quantization method (q4_k_m, f16, etc.)")
    args = parser.parse_args()

    # Debug Environment for Token
    token = os.environ.get("HF_TOKEN")
    if token:
        print(f"DEBUG: HF_TOKEN found {token}")
    else:
        print("DEBUG: HF_TOKEN NOT found in environment variables!")

    if not args.base and not args.base_model_id:
        print("ERROR: Either --base or --base-model-id must be provided")
        sys.exit(1)

    print(f"DEBUG: Starting GGUF conversion for {args.adapter} using local llama.cpp submodule")

    # Locate the vendored script
    # We assume this script runs inside WSL/Linux container relative to project root
    # or that paths are absolute.
    
    # Path to convert_lora_to_gguf.py within the submodule
    # Adjust path: backend/vendor/llama.cpp/convert_lora_to_gguf.py
    # Since we execute this via 'python -m app.training.converter', current dir is usually backend/
    
    script_path = os.path.abspath("backend/vendor/llama.cpp/convert_lora_to_gguf.py")
    
    # Check if script exists, if not, try resolving relative to this file
    if not os.path.exists(script_path):
        current_file_dir = os.path.dirname(os.path.abspath(__file__)) # app/training
        project_root = os.path.dirname(os.path.dirname(current_file_dir)) # backend/
        script_path = os.path.join(project_root, "vendor", "llama.cpp", "convert_lora_to_gguf.py")

    if not os.path.exists(script_path):
        # Fallback: maybe we are running from project root
        script_path = "backend/vendor/llama.cpp/convert_lora_to_gguf.py"
    
    if not os.path.exists(script_path):
        print(f"ERROR: Could not find conversion script at {script_path}")
        sys.exit(1)

    print(f"DEBUG: Using conversion script: {script_path}")
    
    # Construct command
    # python3 script.py --outfile <output> <adapter> [--base <base> | --base-model-id <id>]
    
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
    
    print(f"DEBUG: Executing: {' '.join(cmd)}")
    
    try:
        # Run conversion
        subprocess.check_call(cmd)
        print(f"DEBUG: Conversion complete. GGUF saved to {args.output}")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Conversion failed with code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
