"""Main FastAPI application module."""

import asyncio
import json
import logging
import os
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any, Annotated, Callable

from fastapi import Depends, FastAPI, HTTPException, Request, Response, Header
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from app.models.database import DatabaseManager
from app.plugins.events import bus
from app.plugins.events.persistence import EventStore
from app.plugins.manager import PluginManager
from app.utils.logging_config import setup_logging
from app.models.recording.events import RecordingEvent, RecordingEndRequest
from app.plugins.events.models import EventContext

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

# Initialize static components
event_store = EventStore()
plugin_manager = None
event_bus = None

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
    global event_bus, plugin_manager
    
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

    # Initialize event bus
    logger.info("Initializing event bus")
    event_bus = bus.EventBus()
    await event_bus.start()

    # Initialize plugin manager
    logger.info("Initializing plugin manager")
    plugin_dir = os.path.join(os.path.dirname(__file__), "plugins")
    plugin_manager = PluginManager(plugin_dir, event_bus=event_bus)

    logger.info("Initializing plugins")
    await plugin_manager.discover_plugins()
    await plugin_manager.initialize_plugins()
    
    logger.info("Startup complete")


@app.on_event("shutdown")
async def shutdown() -> None:
    """Clean up application components."""
    global event_bus, plugin_manager
    
    logger.info("Starting application shutdown")
    
    try:
        # First stop accepting new requests
        logger.info("Stopping event bus to prevent new events")
        if event_bus:
            await event_bus.stop()
            
        # Wait for in-flight requests to complete (with timeout)
        logger.info("Waiting for in-flight requests to complete")
        await asyncio.sleep(2)  # Give time for requests to finish
            
        # Then shutdown plugins
        if plugin_manager:
            logger.info("Shutting down plugins")
            await plugin_manager.shutdown_plugins()
            plugin_manager = None

        # Finally shutdown event bus completely
        if event_bus:
            logger.info("Shutting down event bus")
            await event_bus.shutdown()
            event_bus = None
            
        # Close database connections last
        logger.info("Closing database connections")
        db = await DatabaseManager.get_instance()
        await db.close()
        
    except Exception as e:
        logger.error(
            "Error during shutdown",
            extra={
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
        raise
    finally:
        logger.info("Shutdown complete")


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
                    "duration": duration
                },
                "request": {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "path_params": request.path_params,
                    "query_params": dict(request.query_params)
                }
            }
        )
        return response
        
    except asyncio.CancelledError:
        logger.info(
            "Request cancelled during shutdown",
            extra={
                "request_id": request_id,
                "request": {
                    "method": request.method,
                    "url": str(request.url)
                }
            }
        )
        raise
        
    except Exception as e:
        logger.error(
            "Error processing request",
            extra={
                "request_id": request_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "request": {
                    "method": request.method,
                    "url": str(request.url)
                }
            }
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
    logger.error(
        "Validation error",
        extra={
            "errors": exc.errors(),
            "body": exc.body,
        }
    )
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
async def recording_ended(
    request: Request,
    recording_end: RecordingEndRequest,
    x_api_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    """Handle recording ended event."""
    logger.info(
        "Processing recording ended event",
        extra={
            "recording_id": recording_end.recording_id,
            "method": request.method,
            "path": str(request.url),
        }
    )

    # Create event
    event = recording_end.to_event()
    event_id = f"{event.recording_id}_ended_{event.recording_timestamp}"
    event.event_id = event_id
    
    logger.debug(
        "Generated event details",
        extra={
            "recording_id": event.recording_id,
            "event_id": event_id,
            "event_timestamp": event.recording_timestamp,
        }
    )

    # Check for duplicates
    if await event.is_duplicate():
        logger.info(
            "Skipping duplicate recording end event",
            extra={
                "recording_id": event.recording_id,
                "event_id": event_id,
            }
        )
        return {"status": "skipped", "reason": "duplicate_event"}

    # Insert event record
    logger.debug(
        "Inserting event record",
        extra={
            "recording_id": event.recording_id,
            "event_id": event_id,
            "system_audio_path": event.system_audio_path,
            "microphone_audio_path": event.microphone_audio_path,
            "metadata": event.metadata,
        }
    )
    await event.save()

    # Update recording status
    logger.debug(
        "Updating recording status",
        extra={
            "recording_id": event.recording_id,
            "event_id": event_id,
            "status": "completed",
            "system_audio_path": event.system_audio_path,
            "microphone_audio_path": event.microphone_audio_path,
        }
    )
    
    db = DatabaseManager.get_instance()
    await db.execute(
        """
        INSERT INTO recordings (recording_id, status, system_audio_path, microphone_audio_path)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(recording_id) DO UPDATE SET
            status = 'completed',
            system_audio_path = excluded.system_audio_path,
            microphone_audio_path = excluded.microphone_audio_path,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            event.recording_id,
            "completed",
            event.system_audio_path,
            event.microphone_audio_path,
        ),
    )

    # Create and publish event
    logger.debug(
        "Creating RecordingEvent",
        recording_id=event.recording_id,
        event_id=event_id,
        event_data=event.dict(),
    )

    event.set_data()
    
    logger.debug(
        "Publishing event to event bus",
        recording_id=event.recording_id,
        event_id=event_id,
        event_type="RecordingEvent",
        event_data=str(event),
    )
    
    await event_bus.publish(event)

    return {"status": "success", "event_id": event_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8001")),
        reload=True,
    )
