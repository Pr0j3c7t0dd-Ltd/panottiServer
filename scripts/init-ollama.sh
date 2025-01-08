#!/bin/bash
set -e

echo "=== Starting Ollama Initialization ==="

# Wait for Ollama server to be ready
echo "Waiting for Ollama server..."
until curl -s http://ollama:11434/api/version > /dev/null; do
    echo -n "."
    sleep 1
done
echo -e "\nOllama server is ready"

# Check if model exists
echo "Checking for model llama3.1:latest..."
if ! curl -s http://ollama:11434/api/show -d '{"name":"llama3.1:latest"}' | grep -q "llama3.1:latest"; then
    echo "Model not found, pulling llama3.1:latest..."
    # Pull with progress
    curl -X POST http://ollama:11434/api/pull -d '{"name": "llama3.1:latest"}' | while read -r line; do
        echo "$line"
        # Check if this is the last line (contains "digest")
        if echo "$line" | grep -q "digest"; then
            break
        fi
    done
else
    echo "Model llama3.1:latest already exists"
fi

# Verify model is ready
echo "Verifying model..."
until curl -s http://ollama:11434/api/show -d '{"name":"llama3.1:latest"}' | grep -q "llama3.1:latest"; do
    echo -n "."
    sleep 1
done

echo "=== Ollama initialization complete ===" 