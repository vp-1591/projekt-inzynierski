import os
import sys
import json
import shutil
import subprocess
import tempfile


def _strip_bnb_config(base_dir):
    """Create a temp copy of base_dir with quantization_config removed from config.json.

    llama.cpp's convert_lora_to_gguf.py cannot dequantize bitsandbytes weights.
    By stripping the quantization_config, the converter reads the architecture
    metadata without attempting dequantization. The adapter's own safetensors
    contain the actual LoRA weights — the base model.safetensors is only needed
    for tensor shape reference.
    """
    config_path = os.path.join(base_dir, "config.json")
    if not os.path.exists(config_path):
        return base_dir

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if "quantization_config" not in config:
        return base_dir

    tmp_dir = tempfile.mkdtemp(prefix="gguf_base_")
    for entry in os.listdir(base_dir):
        src = os.path.join(base_dir, entry)
        dst = os.path.join(tmp_dir, entry)
        if entry == "config.json":
            clean = {k: v for k, v in config.items() if k != "quantization_config"}
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(clean, f, indent=2)
        else:
            if os.path.isdir(src):
                shutil.copytree(src, dst, symlinks=True)
            else:
                os.symlink(src, dst)

    print(f"Stripped bitsandbytes quantization_config → {tmp_dir}")
    return tmp_dir


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

    if not args.base and not args.base_model_id:
        print("ERROR: Either --base or --base-model-id must be provided")
        sys.exit(1)

    # Locate the vendored llama.cpp conversion script
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

    # If using a local --base dir, strip bitsandbytes config so llama.cpp doesn't crash
    clean_base_dir = None
    if args.base:
        clean_base_dir = _strip_bnb_config(args.base)

    cmd = [
        "python3",
        script_path,
        "--outfile", args.output,
        args.adapter
    ]

    if args.base_model_id:
        cmd.extend(["--base-model-id", args.base_model_id])
    elif clean_base_dir:
        cmd.extend(["--base", clean_base_dir])

    try:
        subprocess.check_call(cmd)
        print(f"Conversion successful. GGUF saved to {args.output}")
    except subprocess.CalledProcessError as e:
        print(f"Conversion failed (code {e.returncode})")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        if clean_base_dir and clean_base_dir != args.base:
            shutil.rmtree(clean_base_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
