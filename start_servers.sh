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

# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Ensure SSL certificate files exist
SSL_KEYFILE="ssl/key.pem"
SSL_CERTFILE="ssl/cert.pem"
if [ ! -f "$SSL_KEYFILE" ] || [ ! -f "$SSL_CERTFILE" ]; then
    echo "SSL certificate files not found. Exiting..."
    exit 1
fi

# Start Admin Frontend server in the background
ADMIN_START_SCRIPT="admin-frontend/start_admin_server.sh"
if [ -f "$ADMIN_START_SCRIPT" ]; then
    echo "Starting Admin Frontend server on port $ADMIN_PORT in the background..."
    lsof -i :"$ADMIN_PORT" -t | xargs kill -9 2>/dev/null  # Kill any process using the port
    chmod +x "$ADMIN_START_SCRIPT"
    (cd admin-frontend && ./start_admin_server.sh) &  # Run in background from admin-frontend directory
else
    echo "Admin frontend start script not found. Exiting..."
    exit 1
fi

# Start FastAPI server in the foreground
echo "Starting FastAPI server on 0.0.0.0:$API_PORT in the foreground..."
lsof -i :"$API_PORT" -t | xargs kill -9 2>/dev/null  # Kill any process using the port
uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "$API_PORT" \
  --reload \
  --ssl-keyfile "$SSL_KEYFILE" \
  --ssl-certfile "$SSL_CERTFILE"
