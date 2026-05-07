import os
import json
import subprocess
import httpx
from datetime import datetime
from sqlalchemy.orm import Session
from ..db import database


def _project_root():
    """Resolve the project root directory (parent of backend/)."""
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == "backend":
        return os.path.dirname(current_dir)
    return current_dir


def _to_wsl(path):
    """Convert a Windows path to a WSL2 path."""
    return path.replace("\\", "/").replace("c:", "/mnt/c").replace("C:", "/mnt/c")


def _get_wsl_host_ip():
    """Resolve the Windows host IP as seen from WSL2 (NAT default gateway).

    Called from Windows because cmd.exe misinterprets pipe chars inside
    subprocess.Popen(shell=True), so the IP must be resolved before
    building the WSL command string.
    """
    result = subprocess.run(
        ['wsl', '--', 'bash', '-c', 'ip route show default | head -1 | cut -d" " -f3'],
        capture_output=True, text=True, timeout=10
    )
    return result.stdout.strip() or "localhost"

class MLOpsOrchestrator:
    """Orchestrates the training, evaluation, and deployment lifecycle of LLM adapters."""
    def __init__(self, db: Session):
        self.db = db
        # Pipeline State
        self.training_progress = 0
        self.evaluation_progress = 0
        self.baseline_f1_non_empty = 0.0
        self.baseline_exact_match = 0.0
        self.new_f1_non_empty = 0.0
        self.new_exact_match = 0.0
        self.status = "idle" # idle, training, evaluating, ready_to_promote
        self.latest_adapter_path = None
        self.on_status_change = [] # Callbacks taking (status_dict)

    def notify(self):
        """Triggers all registered status change callbacks."""
        status = self.get_status()
        for callback in self.on_status_change:
            callback(status)

    def get_status(self):
        """Aggregates and returns the full pipeline status and metrics."""
        baseline = self.read_baseline_metrics()
        return {
            "status": self.status,
            "training_progress": self.training_progress,
            "evaluation_progress": self.evaluation_progress,
            "baseline_f1_non_empty": baseline['f1'],
            "baseline_exact_match": baseline['em'],
            "new_f1_non_empty": self.new_f1_non_empty,
            "new_exact_match": self.new_exact_match
        }

    def read_baseline_metrics(self):
        """Parses the current baseline report to extract F1 and Exact-Match scores."""
        result = {'f1': 0.0, 'em': 0.0}
        try:
            report_path = os.path.join(_project_root(), "model", "benchmark-reports", "current_baseline_report.txt")
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
                
                import re
                match_em = re.search(r"Exact-Match Accuracy: (\d+\.\d+)", content)
                if match_em:
                    result['em'] = float(match_em.group(1))
                    
                match_f1 = re.search(r"Mean Document-Level F1 \(excluding empty gold-label docs\): (\d+\.\d+)", content)
                if not match_f1:
                    match_f1 = re.search(r"Mean F1 \(Non-empty gold docs\): (\d+\.\d+)", content)
                if match_f1:
                    result['f1'] = float(match_f1.group(1))
            return result
        except:
            return result

    def start_manual_training(self, file_path: str):
        """Initiates the training process inside WSL and tracks progress."""
        if self.status not in ["idle", "deployment_success", "deployment_error", "ready_to_promote"]:
            return False
            
        self.status = "training"
        self.training_progress = 0
        self.evaluation_progress = 0
        self.notify()
        
        # Persist run metadata
        new_run = database.TrainingRun(
            status="running",
            start_time=datetime.utcnow()
        )
        self.db.add(new_run)
        self.db.commit()

        # Resolve paths & environment
        wsl_path = file_path.replace("\\", "/").replace("c:", "/mnt/c").replace("C:", "/mnt/c")
        
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{new_run.id}.log")
        
        project_root = _project_root()
        base_model_wsl = _to_wsl(os.path.join(project_root, "model", "bielik-4.5b-base"))

        # Get Host IP for WSL-to-Windows callbacks
        # Resolve the gateway IP on the Windows side — cmd.exe misinterprets pipes
        # inside subprocess.Popen(shell=True), so $(ip route | grep ...) breaks.
        host_ip = _get_wsl_host_ip()

        cmd = f"wsl --exec bash -c \"python3 -u -m app.training.trainer --data {wsl_path} --output ./model/latest --base {base_model_wsl} --backend http://{host_ip}:8000\""
        
        try:
            with open(log_file, "w") as f_log:
                f_log.write(f"--- Training started at {datetime.utcnow()} ---\n")
                f_log.write(f"COMMAND: {cmd}\n\n")
            
            f_log = open(log_file, "a")
            process = subprocess.Popen(
                cmd, 
                shell=True, 
                stdout=f_log, 
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            print(f"Training started (PID: {process.pid}). Logs: {log_file}")
            return True
        except Exception as e:
            print(f"Training launch failed: {str(e)}")
            self.status = "idle"
            new_run.status = "failed"
            self.db.commit()
            self.notify()
            return False

    def update_progress(self, stage: str, value: int):
        """External progress update hook."""
        if stage == "training":
            self.training_progress = value
        elif stage == "evaluation":
            self.evaluation_progress = value
        self.notify()

    def finish_training_and_evaluate(self, adapter_path: str):
        """Transitions the pipeline from training to benchmark evaluation."""
        self.status = "evaluating"
        self.training_progress = 100
        self.evaluation_progress = 0
        self.latest_adapter_path = adapter_path
        self.notify()
        
        import threading
        
        def run_benchmark():
            """Worker thread for running the evaluation benchmark in WSL."""
            try:
                project_root = _project_root()

                # Handle adapter path conversion
                if adapter_path.startswith("/mnt/"):
                    adapter_wsl = adapter_path
                else:
                    adapter_full_win = os.path.normpath(os.path.abspath(os.path.join(project_root, adapter_path) if not os.path.isabs(adapter_path) else adapter_path))
                    adapter_wsl = _to_wsl(adapter_full_win)

                # Prepare WSL command parameters
                base_wsl = _to_wsl(os.path.join(project_root, "model", "bielik-4.5b-base"))
                data_wsl = _to_wsl(os.path.join(project_root, "model", "dataset", "mipd_test.jsonl"))
                output_wsl = _to_wsl(os.path.join(project_root, "model", "benchmark-reports"))
                
                # Get Host IP for WSL-to-Windows callbacks
                host_ip = _get_wsl_host_ip()

                cmd = f"wsl --exec bash -c \"python3 -u -m app.training.benchmark --adapter {adapter_wsl} --base {base_wsl} --data {data_wsl} --backend http://{host_ip}:8000 --output_dir {output_wsl} --no-tqdm\""
                
                # Internal logging
                os.makedirs("logs", exist_ok=True)
                bench_log_file = os.path.join("logs", f"benchmark_{int(datetime.utcnow().timestamp())}.log")
                
                with open(bench_log_file, "w") as f:
                    f.write(f"--- Benchmark started at {datetime.utcnow()} ---\n")
                    f.write(f"Command: {cmd}\n")

                process = subprocess.Popen(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )
                
                captured_f1 = 0.0
                captured_em = 0.0
                
                with open(bench_log_file, "a") as f_bench:
                    while True:
                        line = process.stdout.readline()
                        if not line and process.poll() is not None:
                            break
                        if line:
                            f_bench.write(line)
                            f_bench.flush()
                            
                            # Parse metrics from live output
                            if "FINAL_F1_SCORE:" in line:
                                try: 
                                    captured_f1 = float(line.split(":")[1].strip())
                                    self.new_f1_non_empty = captured_f1
                                except: pass
                            
                            if "FINAL_EXACT_MATCH:" in line:
                                try:
                                    captured_em = float(line.split(":")[1].strip())
                                    self.new_exact_match = captured_em
                                except: pass
                            
                process.wait()
                
                if process.returncode == 0:
                    self.new_f1_non_empty = captured_f1
                    self.status = "ready_to_promote"
                else:
                    self.status = "idle"
                self.notify()
            except Exception as e:
                print(f"Benchmark error: {e}")
                self.status = "idle"
                self.notify()

        threading.Thread(target=run_benchmark).start()

    async def deploy_new_adapter(self, adapter_path: str):
        """Converts the HF adapter to GGUF and hot-swaps the production Ollama model."""
        os.makedirs("logs", exist_ok=True)
        deploy_log_file = os.path.join("logs", f"deploy_{int(datetime.utcnow().timestamp())}.log")
        
        with open(deploy_log_file, "w", encoding="utf-8") as f:
            f.write(f"--- Deployment started at {datetime.utcnow()} ---\n")
            f.write(f"Adapter: {adapter_path}\n")

        def log_deploy(msg):
            with open(deploy_log_file, "a", encoding="utf-8") as f:
                f.write(f"{msg}\n")

        # --- Phase 1: GGUF Conversion ---
        self.status = "deploying" 
        self.notify()
        log_deploy(f"Converting adapter: {adapter_path}")

        project_root = _project_root()
        base_model_wsl = _to_wsl(os.path.join(project_root, "model", "bielik-4.5b-base"))

        inner_cmd = f"python3 -u -m app.training.converter --adapter {adapter_path} --base {base_model_wsl} --output {adapter_path}_gguf --quant_method q4_k_m"
        conversion_cmd = f'wsl --exec bash -c "{inner_cmd}"'
        
        try:
            import asyncio
            process = await asyncio.create_subprocess_shell(
                conversion_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            with open(deploy_log_file, "a", encoding="utf-8") as f_log:
                while True:
                    line = await process.stdout.readline()
                    if not line: break
                    f_log.write(f"{line.decode().strip()}\n")
                    f_log.flush()

            await process.wait()
            if process.returncode != 0:
                 log_deploy(f"Conversion failed (code {process.returncode})")
                 self.status = "ready_to_promote" 
                 self.notify()
                 return False
            log_deploy("Conversion successful.")
        except Exception as e:
            log_deploy(f"Conversion runtime error: {e}")
            self.status = "ready_to_promote"
            self.notify()
            return False

        # --- Phase 2: Hot-Swap ---
        def wsl_to_win(path):
            if path.startswith("/mnt/c/"): return path.replace("/mnt/c/", "c:/")
            if path.startswith("/mnt/d/"): return path.replace("/mnt/d/", "d:/")
            return path
            
        gguf_win_target = wsl_to_win(adapter_path.rstrip("/") + "_gguf")
        
        found_gguf_path = None
        if os.path.isfile(gguf_win_target):
            found_gguf_path = gguf_win_target.replace("\\", "/")
        else:
            try:
                for file in os.listdir(gguf_win_target):
                    if file.endswith(".gguf") or file.endswith("gguf"):
                        found_gguf_path = os.path.join(gguf_win_target, file).replace("\\", "/")
                        break
            except Exception as e:
                print(f"GGUF access error: {e}")
                self.status = "ready_to_promote"
                self.notify()
                return False
            
        if not found_gguf_path:
            self.status = "ready_to_promote"
            self.notify()
            return False

        # Update Modelfile
        modelfile_path = os.path.join(_project_root(), "model", "Modelfile")
        with open(modelfile_path, "r") as f: lines = f.readlines()
        with open(modelfile_path, "w") as f:
            for line in lines:
                f.write(f"ADAPTER {found_gguf_path}\n" if line.startswith("ADAPTER") else line)
        
        # Reload Ollama using CLI
        try:
            self.status = "deploying"
            self.notify()
            create_cmd = ["ollama", "create", "bielik-lora-mipd", "-f", modelfile_path]
            process = subprocess.run(create_cmd, capture_output=True, text=True, encoding='utf-8')
            
            if process.returncode != 0:
                raise Exception(f"Ollama create failed: {process.stderr}")
            
            self.status = "deployment_success"
            self.notify()
        except Exception as e:
            print(f"Ollama hot-swap failed: {e}")
            self.status = "deployment_error"
            self.notify()
            return False
            
        # Update baseline report metadata
        try:
            project_root = _project_root()
            reports_dir = os.path.join(project_root, "model", "benchmark-reports")
            baseline_report_path = os.path.join(reports_dir, "current_baseline_report.txt")
            
            candidates = [os.path.join(reports_dir, f) for f in os.listdir(reports_dir) if f.startswith("benchmark_report_") and f.endswith(".txt")]
            if candidates:
                latest_report = sorted(candidates, key=os.path.getmtime, reverse=True)[0]
                import shutil
                shutil.copy2(latest_report, baseline_report_path)
        except Exception as e:
            print(f"Failed to update baseline metadata: {e}")
            
        return True
