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

# Pull and create the model
echo "Pulling llama3.1 model..."
curl -X POST http://localhost:11434/api/pull -d '{"name":"llama3.1:latest","insecure":true}'

echo "Creating llama3.1-16k model..."
curl -X POST http://localhost:11434/api/create -d '{
    "name":"llama3.1-16k",
    "model":"llama3.1:latest",
    "context_length":16384,
    "num_ctx":16384,
    "num_batch":2048,
    "num_thread":8,
    "repeat_last_n":512,
    "temperature":0.7,
    "top_k":40,
    "top_p":0.9
}'

# Keep the container running
wait 