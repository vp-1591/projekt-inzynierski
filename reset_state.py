import os
import shutil
import sqlite3
import subprocess
import sys

def reset_project_state():
    """
    Resets the Ollama model and benchmark reports to the original xai-adapter baseline.
    """
    # 1. Configuration
    project_root = os.getcwd()
    modelfile_path = os.path.join(project_root, "model", "Modelfile")
    baseline_adapter_path = os.path.join(project_root, "model", "xai-adapter", "checkpoint-2475", "checkpoint-2475-F32-LoRA.gguf")
    
    baseline_report_src = os.path.join(project_root, "model", "benchmark-reports", "xai-adapter", "report.txt")
    baseline_report_dst = os.path.join(project_root, "model", "benchmark-reports", "current_baseline_report.txt")

    print("--- Resetting Project State ---")

    # 2. Check prerequisites
    if not os.path.exists(modelfile_path):
        print(f"Error: Modelfile not found at {modelfile_path}")
        return

    if not os.path.exists(baseline_adapter_path):
        print(f"Error: Baseline adapter not found at {baseline_adapter_path}")
        return

    # 3. Update Modelfile
    print(f"Updating {modelfile_path} to point to xai-adapter...")
    try:
        with open(modelfile_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        with open(modelfile_path, "w", encoding="utf-8") as f:
            for line in lines:
                if line.startswith("ADAPTER"):
                    f.write(f"ADAPTER {baseline_adapter_path}\n")
                else:
                    f.write(line)
        print("Modelfile updated.")
    except Exception as e:
        print(f"Error updating Modelfile: {e}")
        return

    # 4. Recreate Ollama model
    print("Executing 'ollama create' to reset model...")
    try:
        create_cmd = ["ollama", "create", "bielik-lora-mipd", "-f", modelfile_path]
        result = subprocess.run(create_cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            print(f"Ollama Error: {result.stderr}")
            return
        print("Ollama model 'bielik-lora-mipd' successfully reset.")
    except Exception as e:
        print(f"Error running Ollama CLI: {e}")
        return

    # 5. Reset Baseline Report
    print("Resetting benchmark baseline report...")
    try:
        if os.path.exists(baseline_report_src):
            shutil.copy2(baseline_report_src, baseline_report_dst)
            print(f"Baseline report restored from xai-adapter/report.txt")
        else:
            print(f"Warning: Baseline source report not found at {baseline_report_src}")
    except Exception as e:
        print(f"Error restoring report: {e}")

    # 6. Clear stale training run records from the database
    print("Clearing training_runs table...")
    db_path = os.path.join(project_root, "backend", "disinfo_system.db")
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM training_runs")
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            print(f"Cleared {deleted} stale training run(s).")
        else:
            print("Database file not found — skipping (will be created on next app start).")
    except Exception as e:
        print(f"Error clearing database: {e}")

    # 7. Delete latest training artifacts and logs
    print("Deleting temporary training artifacts and logs...")
    try:
        latest_model_path = os.path.join(project_root, "backend", "model", "latest")
        if os.path.exists(latest_model_path):
            shutil.rmtree(latest_model_path)
            print(f"Deleted {latest_model_path}")
            
        backend_logs_path = os.path.join(project_root, "backend", "logs")
        if os.path.exists(backend_logs_path):
            shutil.rmtree(backend_logs_path)
            print(f"Deleted {backend_logs_path}")
    except Exception as e:
        print(f"Error deleting artifacts: {e}")

    print("\n--- Project state successfully reset to xai-adapter baseline ---")
    print("(If the database was missing, it will be auto-created with the correct")
    print(" schema when the backend starts — no manual action needed.)")

if __name__ == "__main__":
    # Ensure we are in the project root (simple heuristic for this project)
    if not os.path.exists("model") or not os.path.exists("backend"):
        print("Error: Please run this script from the project root directory.")
        sys.exit(1)
        
    reset_project_state()
