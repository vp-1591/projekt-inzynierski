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
        
        # NOTE: Pass run_id or some callback URL so arguments can be tracked
        cmd = f"wsl python3 -m backend.app.training.trainer --data {wsl_path} --output ./model/latest"
        
        subprocess.Popen(cmd, shell=True)
        return True

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
        
        # Mock evaluation process for demo (Real implementation would run benchmark script)
        # In real scenario: start benchmark subprocess
        import threading
        import time
        import random
        
        def mock_eval():
            for i in range(0, 101, 10):
                self.evaluation_progress = i
                time.sleep(0.5)
            
            self.new_f1_score = round(random.uniform(0.70, 0.85), 4)
            self.status = "ready_to_promote"
            print(f"DEBUG: Evaluation done. New F1: {self.new_f1_score}")

        threading.Thread(target=mock_eval).start()

    def export_training_data(self):
        # Combine Golden Samples with Replay Buffer
        samples = self.db.query(database.GoldenSample).all()
        
        # Export to JSONL (Alpaca/ChatML format)
        export_path = os.path.join(os.getcwd(), "training_data_latest.jsonl")
        with open(export_path, "w", encoding="utf-8") as f:
            for s in samples:
                # Mocking the output format model expects
                output = json.dumps({
                    "reasoning": s.reasoning,
                    "discovered_techniques": s.expert_tags
                })
                line = {
                    "instruction": "Analyze this text for disinformation techniques...",
                    "input": s.text,
                    "output": output
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        
        return export_path

    async def deploy_new_adapter(self, adapter_path: str):
        """
        Implementation of 2.5.4: Hot-Swap Logic
        """
        # 1. Update the local Modelfile content
        modelfile_path = "c:/Users/vadim/Documents/Vadym/GitRep/projekt-inzynierski/model/Modelfile"
        
        with open(modelfile_path, "r") as f:
            lines = f.readlines()
        
        # Replace the ADAPTER line
        with open(modelfile_path, "w") as f:
            for line in lines:
                if line.startswith("ADAPTER"):
                    f.write(f"ADAPTER {adapter_path}\n")
                else:
                    f.write(line)
        
        # 2. Tell Ollama to recreate the model with new config
        async with httpx.AsyncClient() as client:
            try:
                # Ollama's /api/create endpoint reloads the model from Modelfile
                response = await client.post(
                    "http://localhost:11434/api/create",
                    json={
                        "name": "bielik-lora-mipd",
                        "path": modelfile_path
                    }
                )
                response.raise_for_status()
                print("DEBUG: Ollama model hot-swapped successfully.")
                return True
            except Exception as e:
                print(f"ERROR: Hot-swap failed: {str(e)}")
                return False
