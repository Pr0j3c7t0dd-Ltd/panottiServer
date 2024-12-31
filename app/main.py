import logging
import os
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from .models.database import get_db
from .models.recording.events import RecordingEndRequest, RecordingStartRequest
from .plugins.events import bus
from .plugins.events.persistence import EventStore
from .plugins.manager import PluginManager
from .utils.logging_config import setup_logging

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get configuration from environment
PORT = int(os.getenv("API_PORT", "8001"))

# Initialize event system
event_store: EventStore = EventStore()
event_bus = bus.EventBus(event_store)
plugin_manager = PluginManager("app/plugins", event_bus=event_bus)

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

# HTTP Status Code Constants
HTTP_ERROR_STATUS_CODE = 400


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize application resources"""
    try:
        # Initialize the database
        get_db()
        logger.info("Database initialized")

        # Initialize plugins
        await plugin_manager.discover_plugins()
        await plugin_manager.initialize_plugins()
        app.state.plugin_manager = plugin_manager
        logger.info("Plugins initialized")
    except Exception as e:
        logger.error(f"Error during startup: {e!s}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Cleanup application resources"""
    try:
        if hasattr(app.state, "plugin_manager"):
            await app.state.plugin_manager.shutdown_plugins()
            logger.info("Plugin manager shutdown")
    except Exception as e:
        logger.error(f"Error during shutdown: {e!s}", exc_info=True)
        raise


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Response]
) -> Response:
    """Middleware to log requests with detailed API endpoint information."""
    request_id = request.headers.get("X-Request-ID") or generate_request_id()
    start_time = datetime.utcnow()

    try:
        # Extract and format headers
        headers = dict(request.headers)
        sanitized_headers = headers.copy()
        sensitive_headers = ["authorization", "x-api-key", "cookie"]
        for header in sensitive_headers:
            if header in sanitized_headers:
                sanitized_headers[header] = "[REDACTED]"

        # Log the incoming request with endpoint details
        logger.info(
            "API request received",
            extra={
                "request": {
                    "id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "headers": sanitized_headers,
                    "client": {
                        "host": request.client.host if request.client else "unknown",
                        "port": request.client.port if request.client else "unknown",
                    },
                },
                "endpoint": {
                    "name": request.url.path,
                    "operation": request.method,
                },
            },
        )

        # Process the request and measure timing
        response = await call_next(request)
        duration = (datetime.utcnow() - start_time).total_seconds()

        # Add request ID and timing to response headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = str(duration)

        # Get response headers
        response_headers = dict(response.headers)
        for header in sensitive_headers:
            if header in response_headers:
                response_headers[header] = "[REDACTED]"

        # Log the completed request with metrics
        logger.info(
            "API request completed",
            extra={
                "request_id": request_id,
                "response": {
                    "status_code": response.status_code,
                    "headers": response_headers,
                },
                "metrics": {
                    "duration_seconds": duration,
                    "status_code": response.status_code,
                    "success": response.status_code < HTTP_ERROR_STATUS_CODE,
                },
            },
        )

        return response
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        logger.error(
            "API request failed",
            extra={
                "request_id": request_id,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
                "metrics": {
                    "duration_seconds": duration,
                    "success": False,
                },
            },
            exc_info=True,
        )
        raise


async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header == os.getenv("API_KEY"):
        return api_key_header
    raise HTTPException(status_code=403, detail="Invalid API Key")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle validation errors with detailed information"""
    body = await request.body()
    body_str = body.decode()

    # Format validation errors
    formatted_errors = []
    for error in exc.errors():
        error_dict = {
            "loc": error.get("loc", []),
            "msg": str(error.get("msg", "")),
            "type": error.get("type", ""),
        }
        formatted_errors.append(error_dict)

    # Log the error details
    logger.error(
        "Validation error",
        extra={
            "raw_body": body_str,
            "formatted_errors": formatted_errors,
            "body": exc.body,
        },
    )

    return JSONResponse(
        status_code=422, content={"detail": formatted_errors, "body": exc.body}
    )


@app.get("/api/active-recordings", response_model=list[dict[str, Any]])
async def get_active_recordings() -> list[dict[str, Any]]:
    """Get all active recordings from the database"""
    try:
        # Query database for active recordings
        with get_db() as conn:
            cursor = conn.execute("SELECT * FROM active_recordings")
            recordings = [dict(row) for row in cursor.fetchall()]
            return recordings
    except Exception as e:
        logger.error(f"Error retrieving active recordings: {e!s}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/recording-started", response_model=dict[str, Any])
async def recording_started(
    request: Request,
    event_request: RecordingStartRequest,
    api_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    """Handle a recording started event."""
    try:
        # Create recording event
        recording_event = event_request.to_event()

        # Store in database
        await recording_event.save()

        return {"status": "success", "recording_id": recording_event.recording_id}
    except Exception as e:
        logger.error(f"Error handling recording started event: {e!s}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/recording-ended", response_model=dict[str, Any])
async def recording_ended(
    request: Request,
    event_request: RecordingEndRequest,
    api_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    request_id = request.headers.get("X-Request-ID") or generate_request_id()
    logger.info(
        "Processing recording ended event",
        extra={
            "request_id": request_id,
            "recording_id": event_request.recording_id,
        },
    )

    try:
        # Create recording event
        recording_event = event_request.to_event()

        # Store in database
        await recording_event.save()

        # Publish to event bus
        await event_bus.publish(recording_event)

        logger.info(
            "Successfully processed recording ended event",
            extra={
                "request_id": request_id,
                "recording_id": recording_event.recording_id,
            },
        )

        return {
            "status": "success",
            "recording_id": recording_event.recording_id,
            "request_id": request_id,
        }
    except Exception as e:
        logger.error(
            "Error handling recording ended event",
            extra={
                "request_id": request_id,
                "recording_id": event_request.recording_id,
                "error": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process recording ended event: {e!s}",
        ) from e


if __name__ == "__main__":
    import uvicorn

    PORT = int(os.getenv("API_PORT", "8001"))
    HOST = os.getenv("API_HOST", "127.0.0.1")  # Default to localhost

    uvicorn.run(
        "app.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        ssl_keyfile="ssl/key.pem",
        ssl_certfile="ssl/cert.pem",
    )
