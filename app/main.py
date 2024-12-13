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
import uuid

from .models.event import RecordingEvent, RecordingStartRequest, RecordingEndRequest
from .models.database import get_db, DatabaseManager
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

@app.on_event("startup")
async def startup_event():
    """Initialize application resources"""
    # Initialize the database
    get_db()
    logger.info("Database initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup application resources"""
    # Close database connections
    DatabaseManager.get_instance().close_connections()
    logger.info("Database connections closed")

def generate_request_id():
    return str(uuid.uuid4())

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log requests with detailed header information."""
    print(f"=== log_requests middleware called === URL: {request.url.scheme}://{request.url.netloc}{request.url.path}")  # Debug print with protocol
    request_id = request.headers.get('X-Request-ID') or generate_request_id()
    logger_context = logging.LoggerAdapter(logger, {'request_id': request_id})
    
    # Extract and format headers
    headers = dict(request.headers)
    sanitized_headers = headers.copy()
    sensitive_headers = ['authorization', 'x-api-key', 'cookie']
    for header in sensitive_headers:
        if header in sanitized_headers:
            sanitized_headers[header] = '[REDACTED]'
    
    # Log the request with headers
    logger.info(
        "Incoming request",
        extra={
            'req_headers': sanitized_headers,
            'req_method': request.method,
            'req_path': request.url.path,
            'req_task': "http_middleware",
            'req_id': request_id
        }
    )
    
    # Get request body
    body = await request.body()
    body_content = None
    
    if body:
        try:
            body_content = json.loads(body)
        except json.JSONDecodeError:
            try:
                body_content = body.decode()
            except UnicodeDecodeError:
                body_content = "[BINARY DATA]"
    
    # Process the request
    start_time = datetime.utcnow()
    response = await call_next(request)
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    # Add request ID to response headers
    response.headers['X-Request-ID'] = request_id
    
    # Get response headers
    response_headers = dict(response.headers)
    for header in sensitive_headers:
        if header in response_headers:
            response_headers[header] = '[REDACTED]'
    
    # Log response
    logger.info(
        "Request completed",
        extra={
            "http": {
                "response": {
                    "status_code": response.status_code,
                    "headers": response_headers,
                    "duration_seconds": duration
                }
            },
            'request_id': request_id
        }
    )
    
    return response

async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(
        status_code=403,
        detail="Invalid API Key"
    )

@app.get("/api/active-recordings")
async def get_active_recordings():
    """Get all active recordings from the database"""
    try:
        # Query database for active recordings
        with get_db() as db:
            recordings = db.get_active_recordings()
        return {"status": "success", "recordings": recordings}
    except Exception as e:
        logger.error(f"Error retrieving active recordings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recording-started")
async def recording_started(
    request: Request,
    event_request: RecordingStartRequest,
    api_key: str = Depends(get_api_key)
):
    """Handle a recording started event."""
    try:
        # Log raw request body
        body = await request.body()
        logger.info(
            "Received recording started request",
            extra={
                "raw_body": body.decode(),
                "parsed_request": event_request.model_dump(),
            }
        )
        
        # Convert to event
        event = event_request.to_event()
        logger.info(
            "Converted request to event",
            extra={
                "event_data": event.model_dump()
            }
        )
        
        # Save event
        event.save()
        
        return {"status": "success", "event": event.model_dump()}
    except Exception as e:
        logger.error(f"Error processing recording started event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recording-ended")
async def recording_ended(
    request: Request,
    event_request: RecordingEndRequest,
    api_key: str = Depends(get_api_key)
):
    """Handle a recording ended event."""
    try:
        # Log raw request body
        body = await request.body()
        logger.info(
            "Received recording ended request",
            extra={
                "raw_body": body.decode(),
                "parsed_request": event_request.model_dump(),
            }
        )
        
        # Convert to event
        event = event_request.to_event()
        logger.info(
            "Converted request to event",
            extra={
                "event_data": event.model_dump()
            }
        )
        
        # Save event
        event.save()
        
        return {"status": "success", "event": event.model_dump()}
    except Exception as e:
        logger.error(f"Error processing recording ended event: {str(e)}")
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
