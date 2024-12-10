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

# Store active recordings in memory (replace with proper database in production)
active_recordings: Dict[str, datetime] = {}

@app.post("/api/recording-started")
async def recording_started(
    request: Request,
    event: RecordingStartRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Handle a recording started event.
    
    Args:
        request: Raw request object
        event: Recording started event details
        api_key: API key for authentication
    
    Returns:
        dict: Success message with recording details
    """
    # Log raw request body
    body = await request.body()
    logger.info(
        f"Incoming recording started request with body: {body.decode()}"
    )

    if event.recordingId in active_recordings:
        raise HTTPException(
            status_code=400,
            detail=f"Recording {event.recordingId} is already active"
        )
    
    # Store recording start time
    active_recordings[event.recordingId] = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
    
    logger.info(
        "Recording started",
        extra={
            "recording_id": event.recordingId,
            "timestamp": event.timestamp,
        }
    )
    
    return {
        "status": "success",
        "message": f"Recording {event.recordingId} started",
        "data": {
            "recordingId": event.recordingId,
            "timestamp": event.timestamp
        }
    }

@app.post("/api/recording-ended")
async def recording_ended(
    request: Request,
    event: RecordingEndRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Handle a recording ended event.
    
    Args:
        request: Raw request object
        event: Recording ended event details
        api_key: API key for authentication
    
    Returns:
        dict: Success message with recording details
    """
    # Log raw request body
    body = await request.body()
    logger.info(
        f"Incoming recording ended request with body: {body.decode()}"
    )

    if event.recordingId not in active_recordings:
        raise HTTPException(
            status_code=400,
            detail=f"Recording {event.recordingId} is not active"
        )
    
    # Extract timestamp from recording ID (format: YYYYMMDDHHMMSS_XXXXXXXX)
    recording_start_time = event.recordingId.split('_')[0]
    
    # Validate that file paths use the same timestamp as the recording ID
    expected_prefix = recording_start_time
    if not (event.systemAudioPath.endswith(f"{event.recordingId}_system_audio.wav") and 
            event.MicrophoneAudioPath.endswith(f"{event.recordingId}_microphone.wav")):
        raise HTTPException(
            status_code=400,
            detail=f"File paths must use the recording ID {event.recordingId} in their names"
        )
    
    # Get recording duration
    start_time = active_recordings[event.recordingId]
    end_time = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
    duration = (end_time - start_time).total_seconds()
    
    # Remove from active recordings
    del active_recordings[event.recordingId]
    
    logger.info(
        "Recording ended",
        extra={
            "recording_id": event.recordingId,
            "timestamp": event.timestamp,
            "duration_seconds": duration,
            "system_audio_path": event.systemAudioPath,
            "microphone_audio_path": event.MicrophoneAudioPath
        }
    )
    
    return {
        "status": "success",
        "message": f"Recording {event.recordingId} ended",
        "data": {
            "recordingId": event.recordingId,
            "timestamp": event.timestamp,
            "duration_seconds": duration,
            "systemAudioPath": event.systemAudioPath,
            "MicrophoneAudioPath": event.MicrophoneAudioPath
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        ssl_keyfile="ssl/key.pem",
        ssl_certfile="ssl/cert.pem"
    )
