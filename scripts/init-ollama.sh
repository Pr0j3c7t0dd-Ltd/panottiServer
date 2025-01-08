#!/bin/bash

# Wait for Ollama server to be ready
echo "Waiting for Ollama server..."
until curl -s http://ollama:11434/api/version > /dev/null; do
    sleep 1
done
echo "Ollama server is ready"

# Check if model exists
if ! curl -s http://ollama:11434/api/show -d '{"name":"llama3.1:latest"}' | grep -q "llama3.1:latest"; then
    echo "Pulling llama3.1:latest model..."
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
    sleep 1
done

echo "Ollama initialization complete" 