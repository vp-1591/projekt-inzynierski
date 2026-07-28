import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import threading
from datetime import datetime

import httpx
from sqlalchemy.orm import Session

from ..db import database

# Docker networking: service hostnames resolved via docker-compose DNS
BACKEND_HOST = os.getenv("BACKEND_HOST", "backend")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://ollama:11434")

# ── Ollama model configuration ───────────────────────────────────────
# These constants mirror model/Modelfile.docker.  Ollama v0.5.5+ (Jan 2025)
# removed Modelfile-based input from POST /api/create; deployment now uses
# the JSON-based API with separate blob uploads.
OLLAMA_MODEL_NAME = "bielik-lora-mipd"

# ChatML template — critical for Bielik
CHATML_TEMPLATE = """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

SYSTEM_PROMPT = """Jesteś ekspertem w dziedzinie analizy mediów i lingwistyki, specjalizującym się w wykrywaniu propagandy, manipulacji poznawczej i błędów logicznych w tekstach w języku polskim.

**Twoje zadanie:**
Przeanalizuj dostarczony tekst wejściowy w języku polskim, aby zidentyfikować konkretne techniki manipulacji. Musisz oprzeć swoją analizę wyłącznie na dostarczonym tekście, szukając wzorców, które mają na celu wpłynięcie na opinię czytelnika za pomocą środków irracjonalnych lub zwodniczych.

**Dozwolone kategorie manipulacji:**
Jesteś ściśle ograniczony do klasyfikowania technik w następujących kategoriach. Nie używaj żadnych innych tagów.

1.  **REFERENCE_ERROR**: Cytaty, które nie popierają tezy, są zmyślone lub pochodzą z niewiarygodnych źródeł.
2.  **WHATABOUTISM**: Dyskredytowanie stanowiska oponenta poprzez zarzucanie mu hipokryzji, bez bezpośredniego odparcia jego argumentów.
3.  **STRAWMAN**: Przeinaczenie argumentu oponenta (stworzenie "chochoła"), aby łatwiej go było zaatakować.
4.  **EMOTIONAL_CONTENT**: Używanie języka nasyconego emocjami (strach, gniew, litość, radość) w celu ominięcia racjonalnego, krytycznego myślenia.
5.  **CHERRY_PICKING**: Zatajanie dowodów lub ignorowanie danych, które zaprzeczają argumentowi, przy jednoczesnym przedstawianiu tylko danych potwierdzających.
6.  **FALSE_CAUSE**: Błędne zidentyfikowanie przyczyny zjawiska (np. mylenie korelacji z przyczynowością).
7.  **MISLEADING_CLICKBAIT**: Nagłówki lub wstępy, które sensacyjnie wyolbrzymiają lub fałszywie przedstawiają faktyczną treść tekstu.
8.  **ANECDOTE**: Wykorzystywanie odosobnionych historii osobistych lub pojedynczych przykładów jako ważnego dowodu na ogólny trend lub fakt naukowy.
9.  **LEADING_QUESTIONS**: Pytania sformułowane w sposób sugerujący konkretną odpowiedź lub zawierające nieudowodnione założenie.
10. **EXAGGERATION**: Hiperboliczne stwierdzenia, które wyolbrzymiają fakty, aby wywołać reakcję.
11. **QUOTE_MINING**: Wyrywanie cytatów z kontekstu w celu zniekształcenia intencji pierwotnego autora.

**Format wyjściowy:**
Musisz odpowiedzieć pojedynczym, poprawnym obiektem JSON zawierającym dwa klucze:
1.  `"reasoning"`: Spójny akapit w **języku polskim** wyjaśniający, które techniki znaleziono i dlaczego. Musisz przytoczyć konkretną logikę lub fragmenty tekstu, aby uzasadnić swoją klasyfikację.
2.  `"discovered_techniques"`: Lista ciągów znaków (stringów) zawierająca dokładnie te tagi, które zdefiniowano powyżej. Jeśli nie znaleziono żadnych technik, zwróć pustą listę.

**Przykładowa struktura:**
{
    "reasoning": "Tekst stosuje [Nazwa Techniki], ponieważ autor sugeruje, że...",
    "discovered_techniques": ["NAZWA_TECHNIKI"]
}
"""

MODEL_PARAMETERS = {
    "temperature": 0.1,
    "stop": ["<|im_end|>"],
    "num_ctx": 16384,
}


async def _upload_blob(client: httpx.AsyncClient, file_path: str) -> str:
    """Compute SHA-256 digest of *file_path* and upload it to Ollama.

    If the blob already exists (HTTP 200 from HEAD), the upload is skipped.
    Returns the digest string in the format ``sha256:<hex>``.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    digest = f"sha256:{sha256.hexdigest()}"
    file_size = os.path.getsize(file_path)

    # Check if the blob already exists — skip re-upload for large files.
    head_resp = await client.head(f"{OLLAMA_API_URL}/api/blobs/{digest}")
    if head_resp.status_code == 200:
        logging.info("Blob %s already exists in Ollama, skipping upload", digest[:16])
        return digest

    logging.info("Uploading blob %s (%d bytes) to Ollama", digest[:16], file_size)
    with open(file_path, "rb") as f:
        resp = await client.post(
            f"{OLLAMA_API_URL}/api/blobs/{digest}",
            content=f,
            headers={"Content-Type": "application/octet-stream"},
        )
    if resp.status_code not in (200, 201):
        raise Exception(f"Failed to upload blob {digest[:16]} (HTTP {resp.status_code}): {resp.text[:200]}")
    return digest


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
        "training_error",
        "ready_to_promote",
        "deployment_success",
        "deployment_error",
    }

    def __init__(self, db: Session):
        self.db = db
        self._lock = threading.RLock()
        # Pipeline State
        self.training_progress = 0
        self.evaluation_progress = 0
        self.baseline_f1_non_empty = 0.0
        self.baseline_exact_match = 0.0
        self.new_f1_non_empty = 0.0
        self.new_exact_match = 0.0
        self.status = "idle"  # idle, training, evaluating, ready_to_promote, training_error
        self.latest_adapter_path = None
        self.deployed_adapter_path = None
        self.last_deployment_status = None
        self.current_run_id = None
        self.on_status_change = []  # Callbacks taking (status_dict)

    def notify(self):
        """Triggers all registered status change callbacks."""
        with self._lock:
            status = self.get_status()
        for callback in self.on_status_change:
            callback(status)

    def get_status(self):
        """Aggregates and returns the full pipeline status and metrics."""
        baseline = self.read_baseline_metrics()
        with self._lock:
            return {
                "status": self.status,
                "training_progress": self.training_progress,
                "evaluation_progress": self.evaluation_progress,
                "baseline_f1_non_empty": baseline["f1"],
                "baseline_exact_match": baseline["em"],
                "new_f1_non_empty": self.new_f1_non_empty,
                "new_exact_match": self.new_exact_match,
                "deployed_adapter_path": self.deployed_adapter_path,
                "last_deployment_status": self.last_deployment_status,
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
        result = {"f1": 0.0, "em": 0.0}
        try:
            report_path = os.path.join(_project_root(), "model", "benchmark-reports", "current_baseline_report.txt")
            with open(report_path, encoding="utf-8") as f:
                content = f.read()

                match_em = re.search(r"Exact-Match Accuracy: (\d+\.\d+)", content)
                if match_em:
                    result["em"] = float(match_em.group(1))
                else:
                    logging.warning("Could not parse Exact-Match from baseline report at %s", report_path)

                match_f1 = re.search(r"Mean Document-Level F1 \(excluding empty gold-label docs\): (\d+\.\d+)", content)
                if not match_f1:
                    match_f1 = re.search(r"Mean F1 \(Non-empty gold docs\): (\d+\.\d+)", content)
                if match_f1:
                    result["f1"] = float(match_f1.group(1))
                else:
                    logging.warning("Could not parse F1 from baseline report at %s", report_path)
            return result
        except FileNotFoundError:
            logging.warning("Baseline report not found at %s; returning zeros", report_path)
            return result
        except Exception as e:
            logging.warning("Failed to read baseline metrics: %s; returning zeros", e)
            return result

    def start_manual_training(self, file_path: str):
        """Initiates the training process and tracks progress."""
        with self._lock:
            if self.status not in self.STARTABLE_STATUSES:
                return False

            self.reset_candidate_state()
            self.status = "training"

        self.notify()

        # Persist run metadata
        new_run = database.TrainingRun(status="running", start_time=datetime.utcnow())
        self.db.add(new_run)
        self.db.commit()
        self.db.refresh(new_run)

        with self._lock:
            self.current_run_id = new_run.id

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"training_{new_run.id}.log")

        project_root = _project_root()
        base_model_path = os.path.join(project_root, "model", "bielik-4.5b-base")

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "app.training.trainer",
            "--data",
            file_path,
            "--output",
            "./model/latest",
            "--base",
            base_model_path,
            "--backend",
            BACKEND_URL,
        ]

        try:
            with open(log_file, "w", encoding="utf-8") as f_log:
                f_log.write(f"--- Training started at {datetime.utcnow()} ---\n")
                f_log.write(f"COMMAND: {' '.join(cmd)}\n\n")

            f_log = open(log_file, "a", encoding="utf-8")  # noqa: SIM115
            process = subprocess.Popen(cmd, stdout=f_log, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
            print(f"Training started (PID: {process.pid}). Logs: {log_file}")

            # Monitor subprocess: detect crashes and close leaked log handle
            monitor = threading.Thread(
                target=self._monitor_training_process,
                args=(process, new_run.id, log_file, f_log),
                daemon=True,
            )
            monitor.start()

            return True
        except Exception as e:
            print(f"Training launch failed: {str(e)}")
            with self._lock:
                self.status = "training_error"
            new_run.status = "failed"
            new_run.end_time = datetime.utcnow()
            self.db.commit()
            self.notify()
            return False

    def _monitor_training_process(self, process, run_id, log_file, log_handle):
        """Daemon thread: waits for training subprocess, transitions state on crash."""
        try:
            process.wait()
            # Close the leaked log handle regardless of outcome
            with contextlib.suppress(Exception):
                log_handle.close()
            if process.returncode != 0:
                with self._lock:
                    self.status = "training_error"
                    self.training_progress = 0
                if run_id:
                    run = self.db.query(database.TrainingRun).get(run_id)
                    if run:
                        run.status = "failed"
                        run.end_time = datetime.utcnow()
                        self.db.commit()
                self.notify()
        except Exception:
            logging.exception("Monitor thread error for run_id=%s", run_id)
            with contextlib.suppress(Exception):
                log_handle.close()
            with self._lock:
                self.status = "training_error"
                self.training_progress = 0
            if run_id:
                with contextlib.suppress(Exception):
                    run = self.db.query(database.TrainingRun).get(run_id)
                    if run:
                        run.status = "failed"
                        run.end_time = datetime.utcnow()
                        self.db.commit()
            with contextlib.suppress(Exception):
                self.notify()

    def update_progress(self, stage: str, value: int):
        """External progress update hook."""
        with self._lock:
            if stage == "training":
                self.training_progress = value
            elif stage == "evaluation":
                self.evaluation_progress = value
        self.notify()

    def finish_training_and_evaluate(self, adapter_path: str):
        """Transitions the pipeline from training to benchmark evaluation."""
        with self._lock:
            if self.status != "training":
                return  # Already transitioned, ignore duplicate
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

                test_dataset_path = os.path.join(project_root, "model", "dataset", "mipd_test.jsonl")

                cmd = [
                    sys.executable,
                    "-u",
                    "-m",
                    "app.training.benchmark",
                    "--adapter",
                    adapter_path,
                    "--base",
                    base_model_path,
                    "--data",
                    test_dataset_path,
                    "--backend",
                    BACKEND_URL,
                ]

                # Internal logging
                os.makedirs("logs", exist_ok=True)
                bench_log_file = os.path.join("logs", f"benchmark_{int(datetime.utcnow().timestamp())}.log")

                with open(bench_log_file, "w", encoding="utf-8") as f:
                    f.write(f"--- Benchmark started at {datetime.utcnow()} ---\n")
                    f.write(f"Command: {' '.join(cmd)}\n")

                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace"
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
                                    with self._lock:
                                        self.new_f1_non_empty = captured_f1
                                except Exception:
                                    pass

                            if "FINAL_EXACT_MATCH:" in line:
                                try:
                                    captured_em = float(line.split(":")[1].strip())
                                    with self._lock:
                                        self.new_exact_match = captured_em
                                except Exception:
                                    pass

                process.wait()

                if process.returncode == 0:
                    with self._lock:
                        self.new_f1_non_empty = captured_f1
                        self.status = "ready_to_promote"
                    if self.current_run_id:
                        run = self.db.query(database.TrainingRun).get(self.current_run_id)
                        if run:
                            run.f1_score_before = self.read_baseline_metrics()["f1"]
                            run.f1_score_after = captured_f1
                            run.end_time = datetime.utcnow()
                            run.status = "ready_to_promote"
                            self.db.commit()
                else:
                    with self._lock:
                        self.status = "training_error"
                    if self.current_run_id:
                        run = self.db.query(database.TrainingRun).get(self.current_run_id)
                        if run:
                            run.end_time = datetime.utcnow()
                            run.status = "failed"
                            self.db.commit()
                self.notify()
            except Exception as e:
                logging.exception("Benchmark error: %s", e)
                with self._lock:
                    self.status = "training_error"
                with contextlib.suppress(Exception):
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
        with self._lock:
            self.status = "deploying"
        self.notify()
        log_deploy(f"Converting adapter: {adapter_path}")

        project_root = _project_root()
        base_model_path = os.path.join(project_root, "model", "bielik-4.5b-base")

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "app.training.converter",
            "--adapter",
            adapter_path,
            "--base",
            base_model_path,
            "--output",
            f"{adapter_path}.gguf",
            "--quant_method",
            "q4_k_m",
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
            )

            with open(deploy_log_file, "a", encoding="utf-8") as f_log:
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        break
                    f_log.write(f"{line.decode('utf-8', errors='replace').strip()}\n")
                    f_log.flush()

            await process.wait()
            if process.returncode != 0:
                log_deploy(f"Conversion failed (code {process.returncode})")
                with self._lock:
                    self.last_deployment_status = "deployment_error"
                    self.status = "ready_to_promote"
                self.notify()
                return False
            log_deploy("Conversion successful.")
        except Exception as e:
            log_deploy(f"Conversion runtime error: {e}")
            with self._lock:
                self.last_deployment_status = "deployment_error"
                self.status = "ready_to_promote"
            self.notify()
            return False

        # --- Phase 2: Hot-Swap ---
        # The converter writes a single .gguf file (not a directory).
        gguf_path = adapter_path.rstrip("/") + ".gguf"

        found_gguf_path = None
        if os.path.isfile(gguf_path):
            found_gguf_path = gguf_path
        else:
            # Fallback: older runs may have used the _gguf suffix without extension
            legacy_path = adapter_path.rstrip("/") + "_gguf"
            if os.path.isfile(legacy_path):
                found_gguf_path = legacy_path
            else:
                # Last resort: search for any .gguf file in the adapter directory
                try:
                    adapter_dir = os.path.dirname(adapter_path)
                    for file in os.listdir(adapter_dir):
                        if file.endswith(".gguf"):
                            found_gguf_path = os.path.join(adapter_dir, file)
                            break
                except Exception as e:
                    print(f"GGUF access error: {e}")
                    with self._lock:
                        self.last_deployment_status = "deployment_error"
                        self.status = "ready_to_promote"
                    self.notify()
                    return False

        if not found_gguf_path:
            with self._lock:
                self.last_deployment_status = "deployment_error"
                self.status = "ready_to_promote"
            self.notify()
            return False

        # Create model in Ollama via JSON-based HTTP API.
        # Ollama v0.5.5+ (Jan 2025) removed Modelfile-based input from
        # POST /api/create.  The new format requires:
        #   1. Upload GGUF files as blobs via POST /api/blobs/{digest}
        #   2. Call POST /api/create with JSON fields (files, adapters,
        #      template, system, parameters) instead of a Modelfile string.
        try:
            with self._lock:
                self.status = "deploying"
            self.notify()

            # Resolve paths for the base model and adapter GGUF.
            # Both containers mount ./model, but at different mount points:
            #   backend → /app/model/    ollama → /model/
            # We read files from the backend mount point; the blob upload
            # sends the raw bytes to Ollama via HTTP, so mount-point
            # differences don't matter for the upload step.
            base_model_dir = os.path.join(project_root, "model", "bielik-4.5b-base")
            base_gguf_filename = None
            base_gguf_path = None
            for fname in os.listdir(base_model_dir):
                if fname.endswith(".gguf"):
                    base_gguf_filename = fname
                    base_gguf_path = os.path.join(base_model_dir, fname)
                    break
            if not base_gguf_path:
                raise FileNotFoundError(f"No .gguf file found in {base_model_dir}")

            adapter_gguf_filename = os.path.basename(found_gguf_path)

            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                # Phase 1: Upload blobs to Ollama
                log_deploy(f"Uploading base model blob: {base_gguf_filename}")
                base_digest = await _upload_blob(client, base_gguf_path)
                log_deploy(f"Base model digest: {base_digest}")

                log_deploy(f"Uploading adapter blob: {adapter_gguf_filename}")
                adapter_digest = await _upload_blob(client, found_gguf_path)
                log_deploy(f"Adapter digest: {adapter_digest}")

                # Phase 2: Create model via JSON-based /api/create
                log_deploy(f"Creating model {OLLAMA_MODEL_NAME} in Ollama via JSON API")

                async with client.stream(
                    "POST",
                    f"{OLLAMA_API_URL}/api/create",
                    json={
                        "name": OLLAMA_MODEL_NAME,
                        "files": {base_gguf_filename: base_digest},
                        "adapters": {adapter_gguf_filename: adapter_digest},
                        "template": CHATML_TEMPLATE,
                        "system": SYSTEM_PROMPT,
                        "parameters": MODEL_PARAMETERS,
                        "stream": True,
                    },
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise Exception(
                            f"Ollama create failed (HTTP {response.status_code}): {body.decode(errors='replace')}"
                        )
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        log_deploy(line)
                        # Detect errors in the streaming response
                        try:
                            chunk = json.loads(line)
                            if "error" in chunk:
                                raise Exception(f"Ollama create failed: {chunk['error']}")
                        except json.JSONDecodeError:
                            pass

            with self._lock:
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
            log_deploy(f"Ollama hot-swap failed: {e}")
            print(f"Ollama hot-swap failed: {e}")
            with self._lock:
                self.status = "deployment_error"
                self.last_deployment_status = "deployment_error"
            self.notify()
            return False

        # Update baseline report metadata
        try:
            project_root = _project_root()
            reports_dir = os.path.join(project_root, "model", "benchmark-reports")
            baseline_report_path = os.path.join(reports_dir, "current_baseline_report.txt")

            candidates = [
                os.path.join(reports_dir, f)
                for f in os.listdir(reports_dir)
                if f.startswith("benchmark_report_") and f.endswith(".txt")
            ]
            if candidates:
                latest_report = sorted(candidates, key=os.path.getmtime, reverse=True)[0]
                import shutil

                shutil.copy2(latest_report, baseline_report_path)
        except Exception as e:
            print(f"Failed to update baseline metadata: {e}")

        # REASON: notify() must fire after the baseline file is updated so the
        # frontend receives fresh metrics. The old polling approach masked this
        # ordering bug; WebSockets only push on explicit notify().
        with self._lock:
            self.new_f1_non_empty = 0.0
            self.new_exact_match = 0.0
        self.notify()

        return True
