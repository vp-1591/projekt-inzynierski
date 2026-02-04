import os
import json
import subprocess
import httpx
from datetime import datetime
from sqlalchemy.orm import Session
from ..db import database

class MLOpsOrchestrator:
    def __init__(self, db: Session):
        self.db = db
        # Runtime state (in-memory for simplicity, use Redis for prod)
        self.training_progress = 0
        self.evaluation_progress = 0
        self.current_f1_score = 0.0
        self.new_f1_score = 0.0
        self.status = "idle" # idle, training, evaluating, ready_to_promote
        self.latest_adapter_path = None

    def get_status(self):
        return {
            "status": self.status,
            "training_progress": self.training_progress,
            "evaluation_progress": self.evaluation_progress,
            "baseline_f1": self.read_baseline_f1(),
            "new_f1": self.new_f1_score
        }

    def read_baseline_f1(self):
        try:
            report_path = r"c:\Users\vadim\Documents\Vadym\GitRep\projekt-inzynierski\model\benchmark_reports\bielik-4.5b-lora-mipd-report.txt"
            with open(report_path, "r") as f:
                content = f.read()
                # Parse "Global Average F1 Score: 0.7824"
                import re
                match = re.search(r"Global Average F1 Score: (\d+\.\d+)", content)
                return float(match.group(1)) if match else 0.0
        except:
            return 0.0

    def start_manual_training(self, file_path: str):
        if self.status != "idle":
            return False
            
        self.status = "training"
        self.training_progress = 0
        self.evaluation_progress = 0
        
        # 1. Record the run
        new_run = database.TrainingRun(
            status="running",
            start_time=datetime.utcnow()
        )
        self.db.add(new_run)
        self.db.commit()

        # 2. Trigger WSL (Linux) training
        wsl_path = file_path.replace("\\", "/").replace("c:", "/mnt/c").replace("C:", "/mnt/c")
        
        # Logging setup
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{new_run.id}.log")
        
        # Define model path in WSL format (Resolving Project Root)
        current_dir = os.getcwd()
        if os.path.basename(current_dir) == "backend":
            project_root = os.path.dirname(current_dir)
        else:
            project_root = current_dir
            
        base_model_windows = os.path.join(project_root, "model", "bielik-4.5b-base")
        base_model_wsl = base_model_windows.replace("\\", "/").replace("c:", "/mnt/c").replace("C:", "/mnt/c")

        # Dynamically get Host IP for WSL to call back
        import socket
        try:
            # This usually gets the LAN IP (e.g. 192.168.x.x) which is reachable from WSL
            host_ip = socket.gethostbyname(socket.gethostname())
        except:
            host_ip = "127.0.0.1" # Fallback

        cmd = f"wsl --exec python3 -u -m app.training.trainer --data {wsl_path} --output ./model/latest --base {base_model_wsl} --backend http://{host_ip}:8000"
        
        try:
            with open(log_file, "w") as f_log:
                f_log.write(f"--- Training started at {datetime.utcnow()} ---\n")
                f_log.write(f"COMMAND: {cmd}\n\n")
            
            # Open log in append mode for the subprocess
            f_log = open(log_file, "a")
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=f_log, 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            print(f"DEBUG: Training process started with PID {process.pid}. Logs: {log_file}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to start training: {str(e)}")
            self.status = "idle"
            new_run.status = "failed"
            self.db.commit()
            return False

    def update_progress(self, stage: str, value: int):
        if stage == "training":
            self.training_progress = value
        elif stage == "evaluation":
            self.evaluation_progress = value

    def finish_training_and_evaluate(self, adapter_path: str):
        self.status = "evaluating"
        self.training_progress = 100
        self.evaluation_progress = 0
        self.latest_adapter_path = adapter_path
        
        # Real implementation: run benchmark script via WSL
        import threading
        
        def run_benchmark():
            try:
                # 1. Resolve paths
                current_dir = os.getcwd()
                if os.path.basename(current_dir) == "backend":
                    project_root = os.path.dirname(current_dir)
                else:
                    project_root = current_dir
                    
                # WSL Path Converter
                def to_wsl(path):
                    return path.replace("\\", "/").replace("c:", "/mnt/c").replace("C:", "/mnt/c")
                
                # Handle adapter path (could be Windows path OR WSL path from trainer)
                if adapter_path.startswith("/mnt/"):
                    adapter_wsl = adapter_path
                else:
                    # Assume relative or absolute Windows path
                    if not os.path.isabs(adapter_path):
                         adapter_full_win = os.path.join(project_root, adapter_path)
                    else:
                        adapter_full_win = adapter_path
                    
                    # Normalize and convert
                    adapter_full_win = os.path.normpath(os.path.abspath(adapter_full_win))
                    adapter_wsl = to_wsl(adapter_full_win)

                base_dir_win = os.path.join(project_root, "model", "bielik-4.5b-base")
                dataset_win = os.path.join(project_root, "model", "dataset", "mipd_test.jsonl")
                output_dir_win = os.path.join(project_root, "model", "benchmark_reports")
                
                base_wsl = to_wsl(base_dir_win)
                data_wsl = to_wsl(dataset_win)
                output_wsl = to_wsl(output_dir_win)
                
                # Backend IP
                import socket
                try:
                    host_ip = socket.gethostbyname(socket.gethostname())
                except:
                    host_ip = "127.0.0.1"
                    
                cmd = f"wsl --exec python3 -u -m app.training.benchmark --adapter {adapter_wsl} --base {base_wsl} --data {data_wsl} --backend http://{host_ip}:8000 --output_dir {output_wsl} --no-tqdm"
                
                print(f"DEBUG: Starting benchmark with command: {cmd}")
                
                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                # Stream output to capture F1
                captured_f1 = 0.0
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        print(f"BENCHMARK: {line.strip()}")
                        if "FINAL_F1_SCORE:" in line:
                            try:
                                captured_f1 = float(line.split(":")[1].strip())
                            except:
                                pass
                            
                process.wait()
                
                if process.returncode == 0:
                    self.new_f1_score = captured_f1
                    self.status = "ready_to_promote"
                    print(f"DEBUG: Evaluation done. New F1: {self.new_f1_score}")
                else:
                    print(f"ERROR: Benchmark failed with return code {process.returncode}")
                    self.status = "idle" # Reset to idle on failure
            except Exception as e:
                print(f"ERROR: Benchmark thread failed: {e}")
                self.status = "idle"

        threading.Thread(target=run_benchmark).start()


    async def deploy_new_adapter(self, adapter_path: str):
        """
        Implementation of 2.5.4: Hot-Swap Logic
        """
        # 1. Infer GGUF path from HF adapter path
        # adapter_path comes as WSL path (e.g. /mnt/c/Users/.../model/latest/adapter)
        # We need to find .../adapter_gguf/....gguf
        
        # Helper to convert WSL path back to Windows for Ollama (running on Windows)
        def wsl_to_win(path):
            if path.startswith("/mnt/c/"):
                return path.replace("/mnt/c/", "c:/")
            elif path.startswith("/mnt/d/"):
                return path.replace("/mnt/d/", "d:/")
            return path
            
        hf_wsl_path = adapter_path.rstrip("/")
        gguf_wsl_dir = hf_wsl_path + "_gguf"
        
        # We need to find the actual .gguf file inside that directory
        # Since the backend is running on Windows, we can access these files via Windows paths
        gguf_win_dir = wsl_to_win(gguf_wsl_dir)
        
        print(f"DEBUG: Looking for GGUF in {gguf_win_dir}")
        
        found_gguf_path = None
        try:
            for file in os.listdir(gguf_win_dir):
                if file.endswith(".gguf"):
                    found_gguf_path = os.path.join(gguf_win_dir, file)
                    break
        except Exception as e:
            print(f"ERROR: Could not list GGUF directory: {e}")
            return False
            
        if not found_gguf_path:
            print("ERROR: No .gguf file found in adapter directory")
            return False
            
        # Normalize slashes for Modelfile
        found_gguf_path = found_gguf_path.replace("\\", "/")
        print(f"DEBUG: Found GGUF adapter: {found_gguf_path}")

        # 2. Update the local Modelfile content
        modelfile_path = "c:/Users/vadim/Documents/Vadym/GitRep/projekt-inzynierski/model/Modelfile"
        
        with open(modelfile_path, "r") as f:
            lines = f.readlines()
        
        # Replace the ADAPTER line
        with open(modelfile_path, "w") as f:
            for line in lines:
                if line.startswith("ADAPTER"):
                    f.write(f"ADAPTER {found_gguf_path}\n")
                else:
                    f.write(line)
        
        # 3. Tell Ollama to recreate the model with new config
        async with httpx.AsyncClient() as client:
            try:
                # Ollama's /api/create endpoint reloads the model from Modelfile
                print(f"DEBUG: Sending create request to Ollama for {modelfile_path}")
                response = await client.post(
                    "http://localhost:11434/api/create",
                    json={
                        "name": "bielik-lora-mipd",
                        "path": modelfile_path
                    },
                    timeout=120.0 # Increased timeout for loading
                )
                response.raise_for_status()
                print("DEBUG: Ollama model hot-swapped successfully.")
                return True
            except Exception as e:
                print(f"ERROR: Hot-swap failed: {str(e)}")
                return False
