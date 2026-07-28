#!/bin/sh
set -e

# Start Ollama server in background
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to become ready
echo "Waiting for Ollama server to start..."
attempt=0
max_attempts=60
while [ $attempt -lt $max_attempts ]; do
  if ollama list > /dev/null 2>&1; then
    echo "Ollama server is ready."
    break
  fi
  attempt=$((attempt + 1))
  sleep 2
done

if [ $attempt -eq $max_attempts ]; then
  echo "ERROR: Ollama server failed to start within 120 seconds"
  exit 1
fi

# Create the model if it doesn't already exist
if ollama list | grep -q "bielik-lora-mipd"; then
  echo "Model bielik-lora-mipd already exists, skipping creation."
else
  echo "Creating bielik-lora-mipd model..."
  ollama create bielik-lora-mipd -f /model/Modelfile.docker
  echo "Model created successfully."
fi

# Keep the Ollama server running in the foreground
wait $OLLAMA_PID