import asyncio
import os
import re
import subprocess
import sys
from datetime import datetime

from sqlalchemy.orm import Session

from ..db import database

# Docker networking: service hostnames resolved via docker-compose DNS
BACKEND_HOST = os.getenv("BACKEND_HOST", "backend")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://ollama:11434")


def _project_root():
    """Resolve the project root directory (parent of backend/)."""
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == "backend":
        return os.path.dirname(current_dir)
    return current_dir


class MLOpsOrchestrator:
    """Orchestrates the training, evaluation, and deployment lifecycle of LLM adapters."""
    STARTABLE_STATUSES = {
        "idle",
        "ready_to_promote",
        "deployment_success",
        "deployment_error",
    }

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
        self.deployed_adapter_path = None
        self.last_deployment_status = None
        self.current_run_id = None
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
            "new_exact_match": self.new_exact_match,
            "deployed_adapter_path": self.deployed_adapter_path,
            "last_deployment_status": self.last_deployment_status
        }

    def reset_candidate_state(self):
        """Clear per-run candidate state before launching a new training job."""
        self.training_progress = 0
        self.evaluation_progress = 0
        self.new_f1_non_empty = 0.0
        self.new_exact_match = 0.0
        self.latest_adapter_path = None
        self.current_run_id = None

    def read_baseline_metrics(self):
        """Parses the current baseline report to extract F1 and Exact-Match scores."""
        result = {'f1': 0.0, 'em': 0.0}
        try:
            report_path = os.path.join(_project_root(), "model", "benchmark-reports", "current_baseline_report.txt")
            with open(report_path, encoding="utf-8") as f:
                content = f.read()

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
        """Initiates the training process and tracks progress."""
        if self.status not in self.STARTABLE_STATUSES:
            return False

        self.reset_candidate_state()
        self.status = "training"
        self.notify()

        # Persist run metadata
        new_run = database.TrainingRun(
            status="running",
            start_time=datetime.utcnow()
        )
        self.db.add(new_run)
        self.db.commit()
        self.db.refresh(new_run)
        self.current_run_id = new_run.id

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{new_run.id}.log")

        project_root = _project_root()
        base_model_path = os.path.join(project_root, "model", "bielik-4.5b-base")

        cmd = [
            sys.executable, "-u", "-m", "app.training.trainer",
            "--data", file_path,
            "--output", "./model/latest",
            "--base", base_model_path,
            "--backend", BACKEND_URL
        ]

        try:
            with open(log_file, "w", encoding="utf-8") as f_log:
                f_log.write(f"--- Training started at {datetime.utcnow()} ---\n")
                f_log.write(f"COMMAND: {' '.join(cmd)}\n\n")

            f_log = open(log_file, "a", encoding="utf-8")
            process = subprocess.Popen(
                cmd,
                stdout=f_log,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                errors="replace"
            )
            print(f"Training started (PID: {process.pid}). Logs: {log_file}")
            return True
        except Exception as e:
            print(f"Training launch failed: {str(e)}")
            self.status = "idle"
            new_run.status = "failed"
            new_run.end_time = datetime.utcnow()
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
            """Worker thread for running the evaluation benchmark."""
            try:
                project_root = _project_root()
                base_model_path = os.path.join(project_root, "model", "bielik-4.5b-base")

                cmd = [
                    sys.executable, "-u", "-m", "app.training.benchmark",
                    "--adapter", adapter_path,
                    "--base", base_model_path,
                    "--backend", BACKEND_URL
                ]

                # Internal logging
                os.makedirs("logs", exist_ok=True)
                bench_log_file = os.path.join("logs", f"benchmark_{int(datetime.utcnow().timestamp())}.log")

                with open(bench_log_file, "w", encoding="utf-8") as f:
                    f.write(f"--- Benchmark started at {datetime.utcnow()} ---\n")
                    f.write(f"Command: {' '.join(cmd)}\n")

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace"
                )

                captured_f1 = 0.0
                captured_em = 0.0

                with open(bench_log_file, "a", encoding="utf-8") as f_bench:
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
                    if self.current_run_id:
                        run = self.db.query(database.TrainingRun).get(self.current_run_id)
                        if run:
                            run.f1_score_before = self.read_baseline_metrics()['f1']
                            run.f1_score_after = captured_f1
                            run.end_time = datetime.utcnow()
                            run.status = "ready_to_promote"
                            self.db.commit()
                else:
                    self.status = "idle"
                    if self.current_run_id:
                        run = self.db.query(database.TrainingRun).get(self.current_run_id)
                        if run:
                            run.end_time = datetime.utcnow()
                            run.status = "failed"
                            self.db.commit()
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
        base_model_path = os.path.join(project_root, "model", "bielik-4.5b-base")

        cmd = [
            sys.executable, "-u", "-m", "app.training.converter",
            "--adapter", adapter_path,
            "--base", base_model_path,
            "--output", f"{adapter_path}_gguf",
            "--quant_method", "q4_k_m"
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )

            with open(deploy_log_file, "a", encoding="utf-8") as f_log:
                while True:
                    line = await process.stdout.readline()
                    if not line: break
                    f_log.write(f"{line.decode('utf-8', errors='replace').strip()}\n")
                    f_log.flush()

            await process.wait()
            if process.returncode != 0:
                 log_deploy(f"Conversion failed (code {process.returncode})")
                 self.last_deployment_status = "deployment_error"
                 self.status = "ready_to_promote"
                 self.notify()
                 return False
            log_deploy("Conversion successful.")
        except Exception as e:
            log_deploy(f"Conversion runtime error: {e}")
            self.last_deployment_status = "deployment_error"
            self.status = "ready_to_promote"
            self.notify()
            return False

        # --- Phase 2: Hot-Swap ---
        gguf_output_dir = adapter_path.rstrip("/") + "_gguf"

        found_gguf_path = None
        if os.path.isfile(gguf_output_dir):
            found_gguf_path = gguf_output_dir
        else:
            try:
                for file in os.listdir(gguf_output_dir):
                    if file.endswith(".gguf") or file.endswith("gguf"):
                        found_gguf_path = os.path.join(gguf_output_dir, file)
                        break
            except Exception as e:
                print(f"GGUF access error: {e}")
                self.last_deployment_status = "deployment_error"
                self.status = "ready_to_promote"
                self.notify()
                return False

        if not found_gguf_path:
            self.last_deployment_status = "deployment_error"
            self.status = "ready_to_promote"
            self.notify()
            return False

        # Create model in Ollama
        try:
            self.status = "deploying"
            self.notify()

            # Read the Modelfile template
            modelfile_path = os.path.join(_project_root(), "model", "Modelfile")
            with open(modelfile_path, encoding="utf-8") as f:
                modelfile_content = f.read()

            # Replace local paths with container paths
            modelfile_content = modelfile_content.replace("FROM ./", "FROM /model/")
            # Update ADAPTER path to point to the newly created GGUF file
            adapter_line = f"ADAPTER {found_gguf_path}"
            modelfile_content = re.sub(r'^ADAPTER\s+\S+', adapter_line, modelfile_content, flags=re.MULTILINE)

            # Write a temporary Modelfile with container paths for ollama create
            docker_modelfile_path = os.path.join(_project_root(), "model", "Modelfile.docker")
            with open(docker_modelfile_path, "w", encoding="utf-8") as f:
                f.write(modelfile_content)

            # Use ollama create via Docker CLI (files are volume-mounted in the container)
            create_cmd = [
                "docker", "exec", "ollama-service",
                "ollama", "create", "bielik-lora-mipd",
                "-f", "/model/Modelfile.docker"
            ]
            result = subprocess.run(
                create_cmd,
                capture_output=True, text=True, encoding="utf-8", timeout=120
            )
            if result.returncode != 0:
                raise Exception(f"Ollama create failed: {result.stderr}")

            self.status = "deployment_success"
            self.last_deployment_status = "deployment_success"
            self.deployed_adapter_path = adapter_path
            if self.current_run_id:
                run = self.db.query(database.TrainingRun).get(self.current_run_id)
                if run:
                    run.status = "deployed"
                    run.end_time = datetime.utcnow()
                    self.db.commit()
        except Exception as e:
            print(f"Ollama hot-swap failed: {e}")
            self.status = "deployment_error"
            self.last_deployment_status = "deployment_error"
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

        # REASON: notify() must fire after the baseline file is updated so the
        # frontend receives fresh metrics. The old polling approach masked this
        # ordering bug; WebSockets only push on explicit notify().
        self.new_f1_non_empty = 0.0
        self.new_exact_match = 0.0
        self.notify()

        return True