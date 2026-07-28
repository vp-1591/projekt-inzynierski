# ============================================================
# Stage 1: Builder — install Python dependencies with gcc/g++
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for compiling C extensions (torch, bitsandbytes, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    ca-certificates gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
       | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# --- Layer 1: Runtime dependencies (fastapi, uvicorn, etc.) ---
# These rarely change and should be cached aggressively.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# --- Layer 2: Training dependencies (torch, unsloth, etc.) ---
# These are large (~5 GB) and change less often than application code.
# Splitting them into a separate layer means runtime-only dep changes
# don't invalidate the training layer cache.
COPY requirements-wsl.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-wsl.txt

# ============================================================
# Stage 2: Runtime — slim image with only installed packages + Docker CLI
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install Docker CLI + gcc/g++ (needed by Triton for JIT GPU kernel compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    ca-certificates gnupg curl \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg \
       | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
    && chmod a+r /etc/apt/keyrings/docker.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
       https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
       > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages/ /usr/local/lib/python3.11/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy application code
# NOTE: vendor/llama.cpp is required at runtime for GGUF conversion.
# The .dockerignore excludes most of the submodule (C/C++ source, tests, etc.)
# keeping only convert_lora_to_gguf.py, convert_hf_to_gguf.py, and gguf-py/.
COPY . .

EXPOSE 8000

# --ws websockets is required for WebSocket support
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets"]

# ============================================================
# Stage 3: Dev — hot-reload + remote debugging
# Decision: docs/adr/0001-docker-dev-env-hot-reload.md
# ============================================================
FROM runtime AS dev

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir debugpy watchfiles

EXPOSE 8000 5678

# --reload watches for source changes; --reload-dir limits watching to app code only.
# debugpy listens on 5678 for optional remote debugging (no --wait-for-client).
ENV PYTHONUNBUFFERED=1
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets", "--reload", "--reload-dir", "/app/app"]