# Docker Setup — Disinformation Detection System

## Prerequisites
- Docker Desktop with WSL2 backend enabled (Windows) or Docker Engine (Linux)
- NVIDIA GPU drivers + NVIDIA Container Toolkit (for Ollama and model training)
- At least 8 GB free disk space for Docker images (training image is ~5-8 GB with PyTorch)

## Quick Start
```bash
docker-compose up
```

This starts three services:
- **Ollama** at http://localhost:11435 (internal: 11434) — automatically creates the `bielik-lora-mipd` model on first start
- **Backend API** at http://localhost:8000 (Swagger docs at /docs) — starts after Ollama is healthy and model is ready
- **Frontend** at http://localhost:5173

No manual model creation step is needed. The Ollama container runs an entrypoint script that creates the model from `model/Modelfile.docker` on first boot. On subsequent starts, if the model already exists (persisted in the `ollama_data` volume), creation is skipped.

## Port Mappings
| Service | Port | Purpose |
|---------|------|---------|
| Frontend | 5173 | React dev server |
| Backend | 8000 | FastAPI REST + WebSocket |
| Ollama | 11435 | LLM inference API (internal: 11434) |

## Volume Mounts
| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./model` | `/model` (Ollama) | Base model GGUF + adapter files + entrypoint script |
| `./model` | `/app/model` (Backend) | Training datasets, adapters |
| `./backend` | `/app` (Backend) | Live code reload during development |
| `./frontend` | `/app` (Frontend) | Live code reload during development |
| `backend_uploads` | `/app/uploads` | Persistent upload storage |
| `backend_logs` | `/app/logs` | Persistent training logs |
| `ollama_data` | `/root/.ollama` (Ollama) | Persisted models across restarts |
| Docker socket | `/var/run/docker.sock` (Backend) | Allows deployment pipeline to recreate Ollama model |

## GPU Setup

Ollama and training both require GPU access. Ensure:
1. NVIDIA drivers are installed on the host
2. NVIDIA Container Toolkit is installed: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
3. Docker Desktop (Windows) has GPU support enabled in Settings → Resources → WSL Integration

Verify GPU access:
```bash
docker exec -it backend-service nvidia-smi
```

## Architecture

```
Frontend (React :5173) → Backend (FastAPI :8000) → Ollama (:11434) → Bielik LLM
                                    ↓
                            Training Pipeline (Unsloth/QLoRA)
```

In Docker, services communicate via the `app-network` bridge network. The backend reaches Ollama at `http://ollama:11434` using Docker DNS. Training scripts reach the backend at `http://backend:8000` for progress callbacks.

## Frontend Backend URL

All frontend API calls are centralized in `frontend/src/config.js`:
```js
export const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';
```
- **Development**: Falls back to `http://localhost:8000` (direct backend access)
- **Production**: Set `VITE_BACKEND_URL=/api` at build time; nginx proxies `/api/` → backend

## Training

Training runs inside the backend container with GPU access. The training pipeline:
1. Upload `.jsonl` training data via the Expert Mode UI
2. Backend launches `python -m app.training.trainer` as a subprocess
3. Trainer reports progress via HTTP callbacks to `http://backend:8000/training/progress`
4. After training, benchmark evaluates the adapter
5. Promote deploys the adapter: backend writes a new `Modelfile.docker` and runs `docker exec ollama-service ollama create` to hot-swap the model

The deployment step uses Docker CLI (installed in the backend image) with the Docker socket mounted from the host, allowing the backend container to manage the Ollama container's models.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Services won't start | Check `docker-compose logs <service>` for errors |
| Ollama health check fails | The healthcheck waits for the `bielik-lora-mipd` model to exist. Check `docker-compose logs ollama` for model creation errors |
| Model not loaded | Check `docker-compose logs ollama` — the entrypoint script should create it automatically |
| Training fails | Check GPU access: `docker exec backend-service nvidia-smi` |
| WebSocket not connecting | Verify backend started with `--ws websockets` flag |
| Frontend can't reach backend | Check `BACKEND_URL` in `frontend/src/config.js` |
| Port already in use | Stop conflicting services or change ports in `docker-compose.yml`. Ollama uses port 11435 on the host to avoid conflict with a local Ollama on 11434 |
| Deployment (promote) fails | Ensure Docker socket is mounted in backend container and Docker CLI is installed |

## Development vs Production

**Development** (default) — source code is mounted for live reload:
```bash
docker compose up
```
The `docker-compose.yml` targets the `dev` stage of the frontend Dockerfile, which runs Vite's dev server on port 5173. The backend runs uvicorn with source mounted at `/app`.

**Production** — optimized builds with no source mounts:
```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build
```
The production override:
- Builds the frontend as static assets served by nginx on port 80 (SPA + reverse proxy to backend)
- Removes source volume mounts (image contains built assets)
- Sets `VITE_BACKEND_URL=/api` so the frontend uses nginx as a reverse proxy

The frontend Dockerfile has three stages:
1. **dev** — Node dev server with hot reload (used by docker-compose)
2. **build** — Compiles Vite production bundle
3. **production** — nginx:alpine serves static assets and proxies `/api/` and `/ws/` to the backend

The backend Dockerfile uses a multi-stage build:
1. **builder** — Installs all Python dependencies with gcc/g++ and Docker CLI
2. **runtime** — Copies only installed packages and Docker CLI (no build tools), reducing image size by ~200 MB

Both Dockerfiles use BuildKit cache mounts (`--mount=type=cache`) for pip and npm downloads, which persist across builds without bloating the image.