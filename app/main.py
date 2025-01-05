"""Main FastAPI application module."""

import json
import logging
import os
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from .models.database import DatabaseManager
from .plugins.events import bus
from .plugins.events.persistence import EventStore
from .plugins.manager import PluginManager
from .utils.logging_config import setup_logging
from .models.recording.events import RecordingEvent
from .plugins.events.models import EventContext

# Configure logging
logger = logging.getLogger(__name__)

# HTTP Status Codes
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 403

# Initialize FastAPI app
app = FastAPI(
    title="Recording Events API",
    description="API for handling start and end recording events",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
event_bus = bus.EventBus()
event_store = EventStore()
plugin_manager = PluginManager("app/plugins", event_bus=event_bus)

# API key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


async def get_api_key(api_key_header: str = Depends(api_key_header)) -> str:
    """Validate API key from request header.

    Args:
        api_key_header: API key from request header

    Returns:
        Validated API key

    Raises:
        HTTPException: If API key is invalid
    """
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(
        status_code=HTTP_UNAUTHORIZED,
        detail="Could not validate API key",
    )


@app.on_event("startup")
async def startup() -> None:
    """Initialize application state."""
    # Setup logging first
    setup_logging()
    
    # Set log level from environment
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    print(f"Setting log level to: {log_level}")
    logging.basicConfig(level=log_level)

    logger.info("Initializing database")
    
    # Get database instance and test connection
    db = await DatabaseManager.get_instance()
    await db.execute("SELECT 1")

    logger.info("Initializing plugins")
    await plugin_manager.discover_plugins()
    await plugin_manager.initialize_plugins()
    
    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown() -> None:
    """Clean up application components on shutdown."""
    try:
        # Clean up plugins
        await plugin_manager.shutdown_plugins()

        # Close database connections
        db = DatabaseManager.get_instance()
        db.close_connections()  # This returns None, which is fine for cleanup

        logger.info("Shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)
        raise


@app.middleware("http")
async def api_logging_middleware(
    request: Request, call_next: Callable[[Request], Response]
) -> Response:
    """Middleware to log requests with detailed API endpoint information."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start_time = datetime.utcnow()

    try:
        response = await call_next(request)
        duration = (datetime.utcnow() - start_time).total_seconds()

        logger.info(
            "API request completed",
            extra={
                "request_id": request_id,
                "response": {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                },
                "metrics": {
                    "duration_seconds": duration,
                    "status_code": response.status_code,
                    "success": response.status_code < HTTP_BAD_REQUEST,
                },
            },
        )

        return response
    except Exception as e:
        logger.error(
            "Error processing request",
            extra={
                "request_id": request_id,
                "error": str(e),
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
            },
            exc_info=True,
        )
        raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors in request data.

    Args:
        request: FastAPI request object
        exc: Validation error exception

    Returns:
        JSON response with error details
    """
    return JSONResponse(
        status_code=HTTP_BAD_REQUEST,
        content={
            "detail": exc.errors(),
            "body": exc.body,
        },
    )


@app.post("/api/recording-started")
async def recording_started(request: Request) -> dict[str, Any]:
    """Handle recording started event.

    Args:
        request: FastAPI request object

    Returns:
        Response with recording ID and status

    Raises:
        HTTPException: If recording_id is missing
    """
    data = await request.json()
    recording_id = data.get("recording_id")
    if not recording_id:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="Missing recording_id")

    logger.info(
        "Processing recording started event",
        extra={"recording_id": recording_id},
    )

    # Insert into recordings table
    db = DatabaseManager.get_instance()
    await db.execute(
        """
        INSERT INTO recordings (recording_id, status)
        VALUES (?, ?)
        """,
        (recording_id, "active"),
    )

    # Insert into recording_events table
    await db.execute(
        """
        INSERT INTO recording_events (
            recording_id, event_type, event_timestamp,
            system_audio_path, microphone_audio_path, metadata
        ) VALUES (?, ?, datetime('now'), ?, ?, ?)
        """,
        (
            recording_id,
            "recording.started",
            data.get("system_audio_path"),
            data.get("microphone_audio_path"),
            json.dumps(data.get("metadata", {})),
        ),
    )

    # Notify plugins
    event_data = {
        "event": "recording.started",
        "recording_id": recording_id,
        "system_audio_path": data.get("system_audio_path"),
        "microphone_audio_path": data.get("microphone_audio_path"),
        "metadata": data.get("metadata", {}),
        # Include other data from request except event name
        **{k: v for k, v in data.items() if k not in ["event", "name"]}
    }
    await event_bus.publish(event_data)

    return {"status": "success", "recording_id": recording_id}


@app.post("/api/recording-ended")
async def recording_ended(request: Request) -> dict[str, Any]:
    """Handle recording ended event.

    Args:
        request: FastAPI request object

    Returns:
        Response with recording ID and status

    Raises:
        HTTPException: If recording_id is missing
    """
    data = await request.json()
    recording_id = data.get("recordingId") or data.get("recording_id")
    if not recording_id:
        raise HTTPException(status_code=HTTP_BAD_REQUEST, detail="Missing recordingId")

    logger.info(
        "Processing recording ended event",
        extra={"recording_id": recording_id},
    )

    # Structure initial event data properly
    event_data = {
        "recording_id": recording_id,
        "event_type": "recording.ended",
        "current_event": {
            "recording": {
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat(),
                "audio_paths": {
                    "system": data.get("systemAudioPath"),
                    "microphone": data.get("microphoneAudioPath")
                },
                "metadata": data.get("metadata", {})
            }
        },
        "event_history": {}  # Empty at start of chain
    }

    event = RecordingEvent(
        recording_timestamp=datetime.utcnow().isoformat(),
        recording_id=recording_id,
        event="recording.ended",
        data=event_data,
        context=EventContext(
            correlation_id=str(uuid.uuid4()),
            source_plugin="api"
        )
    )
    await event_bus.publish(event)

    return {"status": "success", "recording_id": recording_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8001")),
        reload=True,
    )
