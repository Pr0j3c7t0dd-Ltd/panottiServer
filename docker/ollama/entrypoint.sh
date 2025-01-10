#!/bin/sh
set -e

# Start Ollama in the background
ollama serve &

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/version > /dev/null; do
    sleep 1
done
echo "Ollama is ready!"

# Pull the model
echo "Pulling llama3.1:latest model..."
ollama pull llama3.1:latest

# Keep the container running
wait 