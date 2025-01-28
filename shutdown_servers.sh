#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a
    source .env
    set +a
fi

# Use ports from environment or default values
API_PORT=${API_PORT:-8001}
ADMIN_PORT=${ADMIN_PORT:-54790}

# Function to kill processes by port
kill_process_by_port() {
    local port=$1
    echo "Looking for processes using port $port..."
    pids=$(lsof -i :"$port" -t)
    if [ -z "$pids" ]; then
        echo "No processes found using port $port."
    else
        echo "Killing processes using port $port..."
        echo "$pids" | xargs kill -9
    fi
}

# Shut down the FastAPI server
echo "Shutting down FastAPI server on port $API_PORT..."
kill_process_by_port "$API_PORT"

# Shut down the Admin Frontend server
echo "Shutting down Admin Frontend server on port $ADMIN_PORT..."
kill_process_by_port "$ADMIN_PORT"

echo "All servers shut down."
