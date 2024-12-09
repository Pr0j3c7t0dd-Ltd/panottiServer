#!/bin/bash

# Load the port from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Use the port from environment or default to 8001
PORT=${API_PORT:-8001}

# Start uvicorn with the specified port and SSL certificates
uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload --ssl-keyfile ssl/key.pem --ssl-certfile ssl/cert.pem
