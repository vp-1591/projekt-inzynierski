# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polish-language disinformation detection system using Explainable AI (XAI). Identifies 11 manipulation techniques in media texts via a fine-tuned Bielik-4.5B LLM with LoRA adapters. The system is an engineering thesis project ("praca inżynierska").

## Commands

**Start all services (Docker):** `docker compose up` — launches Ollama, backend, and frontend in containers. See [DOCKER.md](DOCKER.md) for details.

**Manual start (without Docker):**

- Backend: `cd backend && python -m app.main` (FastAPI on :8000)
- Frontend: `cd frontend && npm run dev` (Vite on :5173)

**Backend lint/format/test (use venv — `cd backend` first):**

- Create venv: `cd backend && python -m venv .venv && .venv/Scripts/pip install -r requirements-dev.txt`
- Lint: `cd backend && .venv/Scripts/ruff check app tests`
- Format check: `cd backend && .venv/Scripts/ruff format --check app tests`
- Auto-fix: `cd backend && .venv/Scripts/ruff check --fix app tests && .venv/Scripts/ruff format app tests`
- Unit tests: `cd backend && .venv/Scripts/python -m pytest tests/` (integration tests are skipped unless `--run-integration` is passed)
- Ruff config is in `backend/ruff.toml` — excludes `vendor/` and `.venv/`

**Frontend build/lint:**

- `cd frontend && npm run build`
- `cd frontend && npm run lint`

**Training (requires NVIDIA GPU):**

- **Docker:** Training runs inside the backend container (GPU passthrough). All dependencies are pre-installed via `requirements-wsl.txt` with pinned versions.
- **WSL2 (without Docker):** `cd backend && python3 -m venv .venv-wsl && .venv-wsl/bin/pip install -r requirements-wsl.txt`
- Training is triggered via the Expert Mode UI, not CLI directly

**Ollama model setup:**

- **Docker:** The Ollama container auto-creates `bielik-lora-mipd` on first start using `model/Modelfile.docker`.
- **Manual:** `ollama create bielik-lora-mipd -f ./model/Modelfile`
- The model name used at runtime is `bielik-lora-mipd:latest` (hardcoded in `backend/app/main.py`)

## Architecture

```
Frontend (React/Vite :5173)
  └── WebSocket + REST ──→ Backend (FastAPI :8000)
                               ├── /analyze → Ollama (:11434) → Bielik LLM
                               ├── /ws/training/status → real-time pipeline updates
                               ├── SQLite (disinfo_system.db) → TrainingRun records
                               └── /upload, /train, /promote → MLOps pipeline
```

**Data flow for analysis:** User text → Frontend → FastAPI `/analyze` → Ollama `/api/chat` → LLM response → `llm_processor.normalize_llm_response()` (3-phase healing: JSON parse → schema normalization → tag validation) → structured result to frontend.

**MLOps pipeline (orchestrator.py):** Upload `.jsonl` → trainer.py (Unsloth SFT, runs in Docker container or WSL2) → benchmark.py (evaluation on mipd_test.jsonl) → converter.py (HF→GGUF) → hot-swap via Ollama CLI. Progress is reported through callbacks → WebSocket broadcast.

## Key Design Decisions

- **LLM response healing** (`backend/app/llm_processor.py`): The 4.5B model often outputs malformed JSON. A 3-phase pipeline (JSON parse → regex recovery → fuzzy key/tag normalization) handles this. Any changes to the output schema must update both `VALID_TAGS` and the regex patterns.

- **11 valid technique tags** are defined in both `backend/app/llm_processor.py` (`VALID_TAGS` set) and `frontend/src/services/disinformationDetector.js` (`TECHNIQUE_MAPPING`). These must stay in sync.

- **Ollama Modelfile** (`model/Modelfile`) uses ChatML template (`<|im_start|>`, `<|im_end|>`) — critical for Bielik model. The system prompt defines the exact JSON output format and allowed technique categories.

- **Training runs inside the Docker backend container** (Linux, GPU passthrough). The Dockerfile installs both runtime and training dependencies. For local development without Docker, training requires WSL2 (Unsloth requires Linux). The backend detects `os.name == 'nt'` and warns. The `ProgressCallback` in trainer.py POSTs progress back to the backend at `{BACKEND_URL}/training/progress` (default `http://localhost:8000`, overridden to `http://backend:8000` in Docker).

- **Baseline metrics** are read from `model/benchmark-reports/current_baseline_report.txt` via regex parsing in `orchestrator.py`.

## Environment

- Git submodule: `backend/vendor/llama.cpp` (run `git submodule update --init --recursive`)
- Database: SQLite file `backend/disinfo_system.db`
- Test dataset: `model/dataset/mipd_test.jsonl` (1521 documents)
- Model files: `model/bielik-4.5b-base/` (GGUF base), `model/xai-adapter/` (production LoRA)

## Logs (Docker)

- **Container stdout/stderr:** `docker compose logs backend` / `docker compose logs frontend` / `docker compose logs ollama`
- **Training/benchmark/deploy log files** are inside the backend container at `/app/logs/`. Access them with:
  - `docker compose exec backend ls //app/logs/` (use `//` to avoid Git Bash path mangling)
  - `docker compose exec backend tail -50 //app/logs/training_2.log`
  - Or copy out: `docker cp <backend-container>:/app/logs ./logs-backup`
- Log files are persisted in the `backend_logs` named Docker volume (survives restarts).

## UI Language

The entire UI and system prompts are in Polish. Technique names displayed to users are Polish translations of the English tags (e.g., `STRAWMAN` → "Chochoł (Słomiana kukła)").

## Git Commits

Do not add "Co-Authored-By: Claude ..." or any AI attribution lines to commit messages.

@~/.claude/shared/adr-workflow.md
