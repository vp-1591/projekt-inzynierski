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

# Install only Docker CLI (no build tools) for Ollama model hot-swap
RUN apt-get update && apt-get install -y --no-install-recommends \
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
# --reload removed for production; add back for dev: CMD ["python", "-m", "uvicorn", ... "--reload"]
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets"]