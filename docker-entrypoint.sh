#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/data /app/logs /app/models/whisper

# Check and download Whisper model if needed
if [ ! -d "/app/models/whisper/models--Systran--faster-whisper-base.en" ]; then
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"Systran/faster-whisper-base.en\", local_dir=\"/app/models/whisper\", local_files_only=False)"
fi

# Initialize restart counters
FASTAPI_RESTART_COUNT=0
MAX_RESTARTS=3

# Function to cleanup processes on exit
cleanup() {
    echo "Cleaning up processes..."
    if [ -n "$NEXT_PID" ]; then
        echo "Stopping Next.js (PID: $NEXT_PID)..."
        kill -SIGTERM $NEXT_PID 2>/dev/null || true
    fi
    if [ -n "$UVICORN_PID" ]; then
        echo "Stopping FastAPI (PID: $UVICORN_PID)..."
        kill -SIGTERM $UVICORN_PID 2>/dev/null || true
    fi
    wait
}

# Function to check if a process is running and restart if needed
check_process() {
    local pid=$1
    local name=$2

    if ! kill -0 $pid 2>/dev/null; then
        echo "Process $name (PID: $pid) has died"
        
        # Only attempt restart for FastAPI
        if [ "$name" = "FastAPI" ] && [ $FASTAPI_RESTART_COUNT -lt $MAX_RESTARTS ]; then
            echo "Attempting to restart $name (Attempt $((FASTAPI_RESTART_COUNT + 1)) of $MAX_RESTARTS)..."
            cd /app
            poetry run uvicorn app.main:app \
                --host ${UVICORN_HOST:-0.0.0.0} \
                --port ${API_PORT} \
                --ssl-keyfile ${SSL_KEY_FILE} \
                --ssl-certfile ${SSL_CERT_FILE} \
                --log-level debug \
                --proxy-headers \
                --workers 1 \
                --timeout-keep-alive 3600 \
                --timeout-graceful-shutdown 3600 \
                --limit-max-requests 0 \
                --backlog 2048 &
            UVICORN_PID=$!
            FASTAPI_RESTART_COUNT=$((FASTAPI_RESTART_COUNT + 1))
            sleep 5
            return 0
        fi
        
        # If Next.js dies or max restarts reached, initiate cleanup
        if [ "$name" = "FastAPI" ]; then
            echo "FastAPI failed to start after $FASTAPI_RESTART_COUNT attempts"
        fi
        cleanup
        exit 1
    fi
    return 0
}

# Trap SIGTERM and SIGINT for graceful shutdown
trap cleanup SIGTERM SIGINT

# Start Next.js admin frontend in the background
cd /app/admin-frontend
echo "Starting Next.js admin frontend..."
PORT=54790 HOST=0.0.0.0 npm run start &
NEXT_PID=$!

# Wait a moment to ensure Next.js starts properly
echo "Waiting for Next.js to start..."
sleep 5

# Verify Next.js is running
check_process $NEXT_PID "Next.js"

# Start FastAPI server in the background
cd /app
echo "Starting FastAPI server..."
poetry run uvicorn app.main:app \
  --host ${UVICORN_HOST:-0.0.0.0} \
  --port ${API_PORT} \
  --ssl-keyfile ${SSL_KEY_FILE} \
  --ssl-certfile ${SSL_CERT_FILE} \
  --log-level debug \
  --proxy-headers \
  --workers 1 \
  --timeout-keep-alive 3600 \
  --timeout-graceful-shutdown 3600 \
  --limit-max-requests 0 \
  --backlog 2048 &
UVICORN_PID=$!

# Wait a moment for FastAPI to start
sleep 2

# Verify FastAPI is running
check_process $UVICORN_PID "FastAPI"

echo "Both services started successfully"

# Monitor processes with improved error handling
while true; do
    check_process $NEXT_PID "Next.js" || break
    check_process $UVICORN_PID "FastAPI" || break
    sleep 10
done

# If we get here, both processes have died
cleanup
exit 1