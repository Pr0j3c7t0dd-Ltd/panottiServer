#!/bin/bash
set -e

echo "=== Starting Ollama Initialization ==="

MAX_RETRIES=30
RETRY_COUNT=0

# Wait for Ollama server to be ready
echo "Waiting for Ollama server..."
while ! curl -s http://ollama:11434/api/version > /dev/null; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Error: Ollama server not ready after $MAX_RETRIES attempts"
        exit 1
    fi
    echo -n "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done
echo -e "\nOllama server is ready"

# Reset retry counter
RETRY_COUNT=0

# Check if model exists
echo "Checking for model llama2:latest..."
while ! curl -s http://ollama:11434/api/show -d '{"name":"llama2:latest"}' | grep -q "llama2:latest"; do
    if [ $RETRY_COUNT -eq 0 ]; then
        echo "Model not found, pulling llama2:latest..."
        # Pull with progress
        curl -X POST http://ollama:11434/api/pull -d '{"name": "llama2:latest"}' | while read -r line; do
            echo "$line"
            # Check if this is the last line (contains "digest")
            if echo "$line" | grep -q "digest"; then
                break
            fi
        done
    fi

    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Error: Model not ready after $MAX_RETRIES attempts"
        exit 1
    fi
    echo -n "."
    sleep 2
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

echo "Model llama2:latest is ready"
echo "=== Ollama initialization complete ===" 