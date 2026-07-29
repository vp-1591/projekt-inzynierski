# ADR 0001: Docker development environment with hot-reload

## Context

The backend runs inside a Docker container with source code volume-mounted at `/app`. However, uvicorn starts **without `--reload`**, so every code change requires manually restarting the container (`docker compose restart backend`) or rebuilding the image and running `docker compose up` again. During debugging sessions, this cycle was repeated 3+ times — edit code, rebuild, restart, check logs — making development slow and painful.

Additionally, there is no remote debugging support inside the container, and the only way to see startup errors is through `docker compose logs backend` after the container crashes.

## Decision

Add a `dev` stage to the backend Dockerfile that extends the existing `runtime` stage with:

1. **`watchfiles`** — a fast file watcher that works reliably on Docker Desktop for Windows (WSL2), replacing uvicorn's default watcher (`watchgod`) which misses inotify events across the NTFS→ext4 boundary.
2. **`debugpy`** — enables remote debugging via VS Code's "Python: Remote Attach" on port 5678.
3. **uvicorn `--reload --reload-dir /app/app`** — watches only the application directory (`/app/app`) to avoid triggering reloads from changes to installed packages, uploads, or logs.

The `dev` stage becomes the Dockerfile's default target (last `FROM`). `docker-compose.yml` uses `target: dev`, while `docker-compose.prod.yml` overrides to `target: runtime` — production images stay lean with no debug tools.

A `command:` override in `docker-compose.yml` explicitly sets the full uvicorn invocation with `--reload`, making the dev-mode startup transparent in `docker compose config`. Production compose overrides this with the non-reload command.

Alternatives considered:
- **Volume-mount `site-packages` from a separate container** — overly complex, no clear benefit over installing dev packages in the `dev` stage.
- **Use `watchdog` (watchgod) instead of `watchfiles`** — unreliable on Windows+WSL2; `watchfiles` uses Rust-based polling as a fallback.
- **Run backend outside Docker during development** — would lose parity with the container environment (Ollama DNS, Docker socket, GPU access).

## Constraints

- The `dev` stage must not change the `runtime` stage; production images remain identical.
- Volume mount `./backend:/app` must continue to work — the `dev` stage relies on it for hot-reload.
- Port 5678 (debugpy) must only be exposed in dev mode, never in production.
- The backend must still start correctly if debugpy is installed but no debugger is attached (no `--wait-for-client`).

## Consequences

- **Positive**: Code changes to `backend/app/**/*.py` trigger automatic uvicorn reload — no rebuild or restart needed.
- **Positive**: Remote debugging via debugpy on port 5678 enables breakpoint-level debugging from VS Code.
- **Positive**: `PYTHONUNBUFFERED=1` ensures log output appears immediately in `docker compose logs`.
- **Positive**: Production images remain lean — `target: runtime` skips dev packages.
- **Negative**: The `dev` image is slightly larger (~10 MB for debugpy + watchfiles). Build time increases by ~5 seconds for the pip install step.
- **Negative**: `--reload` adds a small latency overhead on file-change detection (typically <1 second). Acceptable for dev, never used in production.

## Validation

1. `docker compose build backend` succeeds and the `dev` stage is the final target.
2. `docker compose up` starts the backend with `--reload` visible in the startup logs.
3. Editing a Python file in `backend/app/` triggers uvicorn reload output in `docker compose logs -f backend`.
4. `docker compose -f docker-compose.yml -f docker-compose.prod.yml build backend` builds the `runtime` stage without debugpy.
5. Existing backend tests pass: `cd backend && .venv/Scripts/python -m pytest tests/`