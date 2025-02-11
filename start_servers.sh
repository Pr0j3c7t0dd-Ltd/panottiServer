#!/bin/bash

# Activate Python virtual environment
if [ -f ".venv/bin/activate" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment not found. Exiting..."
    exit 1
fi

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
ADMIN_DIR="admin-frontend"
if [ -d "$ADMIN_DIR" ]; then
    echo "Starting Admin Frontend server on port $ADMIN_PORT in the background..."
    lsof -i :"$ADMIN_PORT" -t | xargs kill -9 2>/dev/null  # Kill any process using the port
    
    # Copy .env.local.sample to .env.local if it doesn't exist
    if [ ! -f "$ADMIN_DIR/.env.local" ] && [ -f "$ADMIN_DIR/.env.local.sample" ]; then
        echo "Creating .env.local from sample..."
        cp "$ADMIN_DIR/.env.local.sample" "$ADMIN_DIR/.env.local"
    fi
    
    # Build and start Next.js from the admin-frontend directory
    if [ "${DEV_MODE:-false}" = "true" ]; then
        echo "Starting in development mode..."
        (cd "$ADMIN_DIR" && npx cross-env PORT="$ADMIN_PORT" npm run dev) &
    else
        echo "Building and starting in production mode..."
        (cd "$ADMIN_DIR" && npm run build && npx cross-env PORT="$ADMIN_PORT" npm start) &
    fi
else
    echo "Admin frontend directory not found. Exiting..."
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
