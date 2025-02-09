#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/data /app/logs /app/models/whisper

# Check and download Whisper model if needed
if [ ! -d "/app/models/whisper/models--Systran--faster-whisper-base.en" ]; then
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download(\"Systran/faster-whisper-base.en\", local_dir=\"/app/models/whisper\", local_files_only=False)"
fi

# Start Next.js admin frontend in the background
cd /app/admin-frontend
PORT=54790 HOST=0.0.0.0 npm run start &

# Wait a moment to ensure Next.js starts properly
sleep 5

# Start FastAPI server
cd /app
exec poetry run uvicorn app.main:app \
  --host ${UVICORN_HOST:-0.0.0.0} \
  --port ${API_PORT} \
  --ssl-keyfile ${SSL_KEY_FILE} \
  --ssl-certfile ${SSL_CERT_FILE} \
  --log-level debug \
  --proxy-headers