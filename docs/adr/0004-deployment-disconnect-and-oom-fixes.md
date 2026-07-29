# ADR 0004: Prevent deployment disconnect and OOM during model hot-swap

## Status

active

## Context

During model deployment (the promote step), the backend became completely unresponsive for the entire duration (5–10 minutes). The frontend showed "Brak połączenia" (disconnected), health checks timed out, and WebSocket connections dropped. Three root causes were identified:

1. **Event-loop blocking**: `_upload_blob()` performed synchronous blocking I/O inside an `async def` — SHA-256 of a 4.7 GB base model file computed on the event loop, and the entire file read into a `bytes` object via `f.read()`. Both operations blocked the single asyncio event loop for seconds to minutes, making the server unresponsive.

2. **OOM kill**: `_compute_sha256_and_read` read the 4.7 GB base model entirely into memory as a `bytes` object (`file_bytes = f.read()`), then passed it as `content=file_bytes` to httpx. The container (5.8 GB total memory) was killed by the OOM killer (`OOMKilled: true` in `docker inspect`).

3. **Infrastructure timeouts**: nginx defaulted to 60s `proxy_read_timeout`, which dropped WebSocket connections and long-running HTTP requests. The `/training/promote` endpoint blocked the HTTP response for the entire deployment, keeping the client waiting.

Secondary issues: no WebSocket keepalive (connections dropped during long deploys), and no Docker restart policies (containers didn't recover after crashes).

## Decision

### 1. Streaming upload with async chunk generator (`orchestrator.py`)

Replace the monolithic `_compute_sha256_and_read()` with two separate functions:

- **`_compute_sha256(file_path)`**: Pure sync function that computes SHA-256 using streaming reads (8 KiB buffer, ~8 KiB peak memory). Called via `asyncio.to_thread()` to avoid blocking the event loop.
- **`_upload_blob()`**: Uploads file content via an async generator (`_chunk_generator()`) that reads 1 MiB chunks through `asyncio.to_thread(f.read, _CHUNK_SIZE)`. httpx accepts `content=<async_generator>` and wraps it as `AsyncByteStream` for streaming upload. Peak memory: ~1 MiB per chunk instead of 4.7 GB.

### 2. WebSocket heartbeat (`main.py`)

Add `ConnectionManager._heartbeat_loop()` that broadcasts `{"type": "heartbeat"}` every 25 seconds to all connected WebSocket clients. The heartbeat starts when the first client connects and stops when no clients remain. Frontend (`App.jsx`) ignores heartbeat messages via `if (data.type === 'heartbeat') return;`.

### 3. nginx timeout overrides (`nginx.conf`)

Add `proxy_read_timeout 3600s` and `proxy_send_timeout 3600s` to both `/api/` and `/ws/` locations. `proxy_connect_timeout` stays at 60s since initial connections are fast. One hour covers the longest deployment.

### 4. Non-blocking promote endpoint (`main.py`)

Change `POST /training/promote` to return `202 Accepted` immediately and run `deploy_new_adapter()` via `asyncio.create_task()`. The client receives deployment progress and completion via the existing WebSocket broadcast. The frontend's `handlePromote` already fires-and-forgets the POST.

### 5. Docker restart policies (`docker-compose.yml`)

Add `restart: unless-stopped` to all three services (ollama, backend, frontend). Containers now recover automatically after crashes or Docker daemon restarts.

### 6. Increased httpx timeout (`orchestrator.py`)

Increase `httpx.AsyncClient` timeout from 300s to 900s. SHA-256 of 4.7 GB + upload over Docker bridge network can take 5+ minutes; 900s provides headroom.

Alternatives considered:
- **`aiofiles` for file I/O**: Would require a new dependency; `asyncio.to_thread()` achieves the same without it.
- **Sendfile/splice syscall**: Not available for HTTP uploads to a different host; httpx must read the data.
- **Chunked upload with progress tracking**: Could resume interrupted uploads, but Ollama's blob API requires the full digest upfront, making resume impossible.
- **Increase container memory limit**: Masks the problem rather than fixing it; the streaming approach is correct regardless of container size.

## Constraints

- The async generator (`_chunk_generator`) opens the file on first iteration and closes it after the last chunk. If the generator is not fully consumed (e.g., connection error), Python's `with` statement ensures the file handle is closed when the generator is garbage-collected.
- httpx must receive `content=<async_generator>` — passing `content=<sync_generator>` would still block the event loop during reads.
- The heartbeat interval (25s) must be shorter than nginx's `proxy_read_timeout` (3600s) to keep the connection alive.
- `asyncio.create_task()` in `/training/promote` means deployment errors are caught and stored in `orchestrator.last_deployment_status` but do not propagate to the HTTP response. Clients must rely on WebSocket status updates.

## Consequences

- **Positive**: Backend stays responsive (3–4ms per request) during multi-minute deployments. No more "Brak połączenia" disconnections.
- **Positive**: Memory usage during blob upload capped at ~1 MiB regardless of file size. OOM kills eliminated.
- **Positive**: WebSocket connections survive long deployments via heartbeat keepalive.
- **Positive**: nginx no longer drops connections after 60s.
- **Positive**: `/training/promote` returns immediately (202), improving UX — the frontend button updates via WebSocket.
- **Positive**: Containers recover automatically after crashes.
- **Negative**: Deployment errors are asynchronous — the HTTP client only knows the task started, not whether it succeeded. The WebSocket must be monitored for completion.
- **Negative**: Async generator upload requires httpx to support `AsyncByteStream` — not all HTTP clients support streaming request bodies.

## Validation

1. All 111 backend unit tests pass (7 new tests: `_compute_sha256` correctness, empty file, `_upload_blob` streaming, skip-on-existing, `asyncio.to_thread` usage).
2. ruff check: all checks passed.
3. Frontend build: ✓ (808ms).
4. **End-to-end Docker test**: Full deployment (convert adapter → SHA-256 → upload 4.7 GB base model blob → upload 200 MB adapter blob → create model in Ollama via JSON API) completed in ~6 minutes. Backend responded to `/docs` in 3–4ms throughout. No OOM. Model `bielik-lora-mipd:latest` confirmed active in Ollama.