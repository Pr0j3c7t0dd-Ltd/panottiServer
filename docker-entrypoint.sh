#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/data /app/logs /app/models/whisper

# Check and download Whisper model if needed
if [ ! -d "/app/models/whisper/models--Systran--faster-whisper-base.en" ]; then
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"Systran/faster-whisper-base.en\", local_dir=\"/app/models/whisper\", local_files_only=False)"
fi

# Function to cleanup processes on exit
cleanup() {
    echo "Cleaning up processes..."
    kill $(jobs -p) 2>/dev/null || true
}

# Trap SIGTERM and SIGINT
trap cleanup SIGTERM SIGINT

# Start Next.js admin frontend in the background
cd /app/admin-frontend
echo "Starting Next.js admin frontend..."
PORT=54790 HOST=0.0.0.0 npm run start &

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
  --timeout-graceful-shutdown 3600 &

# Wait forever
while true; do
    sleep 86400 # Sleep for a day
done