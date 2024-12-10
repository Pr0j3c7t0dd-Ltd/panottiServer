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
from .models.database import get_db
from .utils.logging_config import setup_logging
from .plugins.manager import plugin_manager

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Setup plugin system
plugin_manager.setup()

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

@app.on_event("startup")
async def startup_event():
    """Initialize application resources"""
    # Initialize the database
    get_db()
    logger.info("Database initialized")
    
    # Initialize plugins
    await plugin_manager.call_hook("on_startup", app=app)
    logger.info("Plugins initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup application resources"""
    # Close database connections
    get_db().close_connections()
    logger.info("Database connections closed")
    
    # Cleanup plugins
    await plugin_manager.call_hook("on_shutdown", app=app)
    logger.info("Plugins unloaded")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests."""
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

@app.middleware("http")
async def plugin_middleware(request: Request, call_next):
    """Middleware to run plugin hooks before and after requests."""
    # Run before_request hooks
    await plugin_manager.call_hook("before_request", request=request)
    
    # Process the request
    response = await call_next(request)
    
    # Run after_request hooks
    await plugin_manager.call_hook("after_request", response=response)
    
    return response

async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Invalid API Key"
    )

async def get_active_recordings():
    """Get all active recordings from the database"""
    with get_db().get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT DISTINCT json_extract(data, '$.recordingId') as recording_id,
                   json_extract(data, '$.timestamp') as timestamp
            FROM events 
            WHERE type = 'Recording Started'
            AND recording_id NOT IN (
                SELECT DISTINCT json_extract(data, '$.recordingId')
                FROM events
                WHERE type = 'Recording Ended'
            )
        ''')
        return {row['recording_id']: datetime.fromisoformat(row['timestamp']) 
                for row in cursor.fetchall()}

@app.post("/recording/started")
async def recording_started(
    request: Request,
    event: RecordingStartRequest,
    api_key: str = Depends(get_api_key)
):
    """Handle a recording started event."""
    try:
        # Save event to database
        event.save()
        
        # Run plugin hooks
        await plugin_manager.call_hook("before_recording_start", recording_id=event.recordingId)
        
        return {"status": "success", "message": "Recording started"}
    except Exception as e:
        logger.error(f"Error processing recording start: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recording/ended")
async def recording_ended(
    request: Request,
    event: RecordingEndRequest,
    api_key: str = Depends(get_api_key)
):
    """Handle a recording ended event."""
    try:
        # Save event to database
        event.save()
        
        # Run plugin hooks
        await plugin_manager.call_hook("after_recording_end", recording_id=event.recordingId)
        
        return {"status": "success", "message": "Recording ended"}
    except Exception as e:
        logger.error(f"Error processing recording end: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
