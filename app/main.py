from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from datetime import datetime
import logging
import json
from typing import Dict
import asyncio

from .models.event import RecordingEvent, RecordingStartRequest, RecordingEndRequest
from .utils.logging_config import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get configuration from environment
PORT = int(os.getenv("API_PORT", "8001"))

app = FastAPI(
    title="Recording Events API",
    description="API for handling start and end recording events",
    version="1.0.0",
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Log request details
    body = await request.body()
    logger.debug(
        "Incoming request",
        extra={
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
            "payload": body.decode() if body else None,
            "client_host": request.client.host if request.client else None,
        }
    )
    
    # Process the request
    response = await call_next(request)
    
    # Return response
    return response

async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Invalid API Key"
    )

# Store active sessions in memory (replace with proper database in production)
active_sessions: Dict[str, datetime] = {}

@app.post("/start-recording")
async def start_recording(
    event: RecordingEvent,
    api_key: str = Depends(get_api_key)
):
    """
    Start a recording session.
    
    Args:
        event: Recording event details including session_id and timestamp
        api_key: API key for authentication
    
    Returns:
        dict: Success message with session details
    """
    if event.session_id in active_sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Session {event.session_id} is already active"
        )
    
    active_sessions[event.session_id] = event.timestamp
    
    logger.info(
        "Recording started",
        extra={
            "session_id": event.session_id,
            "timestamp": event.timestamp.isoformat(),
        }
    )
    
    return {
        "status": "success",
        "message": f"Recording started for session {event.session_id}",
        "timestamp": event.timestamp
    }

@app.post("/end-recording")
async def end_recording(
    event: RecordingEvent,
    api_key: str = Depends(get_api_key)
):
    """
    End a recording session.
    
    Args:
        event: Recording event details including session_id and timestamp
        api_key: API key for authentication
    
    Returns:
        dict: Success message with session details
    """
    if event.session_id not in active_sessions:
        raise HTTPException(
            status_code=400,
            detail=f"Session {event.session_id} is not active"
        )
    
    start_time = active_sessions.pop(event.session_id)
    duration = (event.timestamp - start_time).total_seconds()
    
    logger.info(
        "Recording ended",
        extra={
            "session_id": event.session_id,
            "timestamp": event.timestamp.isoformat(),
            "duration_seconds": duration
        }
    )
    
    return {
        "status": "success",
        "message": f"Recording ended for session {event.session_id}",
        "timestamp": event.timestamp,
        "duration_seconds": duration
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=True)
