"""Main FastAPI application module."""

import asyncio
import logging
import os
import traceback
import uuid
import weakref
from collections.abc import Callable
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader

from app.core.events import EventStore
from app.core.events.bus import EventBus
from app.models.database import DatabaseManager
from app.models.recording.events import (
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
)
from app.plugins.manager import PluginManager
from app.utils.directory_sync import DirectorySync
from app.utils.logging_config import setup_logging

# Configure logging
logger = logging.getLogger(__name__)

# HTTP Status Codes
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 403
HTTP_UNPROCESSABLE_ENTITY = 422

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application startup and shutdown."""
    # Startup
    global event_bus, plugin_manager, directory_sync

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

    # Initialize directory sync
    logger.info("Initializing directory sync")
    app_root = Path(__file__).parent.parent
    directory_sync = DirectorySync(app_root)
    directory_sync.start_monitoring()

    # Initialize event bus
    logger.info("Initializing event bus")
    event_bus = EventBus()
    await event_bus.start()

    # Initialize plugin manager
    logger.info("Initializing plugin manager")
    plugin_dir = os.path.join(os.path.dirname(__file__), "plugins")
    plugin_manager = PluginManager(plugin_dir, event_bus=event_bus)

    logger.info("Initializing plugins")
    await plugin_manager.discover_plugins()
    await plugin_manager.initialize_plugins()

    logger.info("Startup complete")

    try:
        yield
    finally:
        # Shutdown
        logger.info("Starting application shutdown")

        try:
            # Cancel all background tasks
            tasks = list(background_tasks)
            if tasks:
                logger.info(f"Cancelling {len(tasks)} background tasks")
                for task in tasks:
                    if not task.done():
                        task.cancel()

                # Wait for tasks to complete with timeout
                try:
                    await asyncio.wait(tasks, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Some tasks did not complete within timeout")

            # Stop directory sync
            if directory_sync:
                logger.info("Stopping directory sync")
                directory_sync.stop_monitoring()
                directory_sync = None

            # First stop accepting new requests
            logger.info("Stopping event bus to prevent new events")
            if event_bus:
                await event_bus.stop()

            # Then shutdown plugins
            if plugin_manager:
                logger.info("Shutting down plugins")
                try:
                    await asyncio.shield(
                        asyncio.wait_for(plugin_manager.shutdown_plugins(), timeout=5.0)
                    )
                except asyncio.TimeoutError:
                    logger.warning("Plugin shutdown timed out after 5 seconds")
                except asyncio.CancelledError:
                    logger.warning("Plugin shutdown cancelled")
                plugin_manager = None

            # Finally shutdown event bus completely
            if event_bus:
                logger.info("Shutting down event bus")
                try:
                    await asyncio.shield(
                        asyncio.wait_for(event_bus.shutdown(), timeout=5.0)
                    )
                except asyncio.TimeoutError:
                    logger.warning("Event bus shutdown timed out after 5 seconds")
                except asyncio.CancelledError:
                    logger.warning("Event bus shutdown cancelled")
                event_bus = None

            # Close database connections last
            logger.info("Closing database connections")
            try:
                db = await DatabaseManager.get_instance()
                await asyncio.shield(asyncio.wait_for(db.close(), timeout=5.0))
            except asyncio.TimeoutError:
                logger.warning("Database shutdown timed out after 5 seconds")
            except asyncio.CancelledError:
                logger.warning("Database shutdown cancelled")
            except Exception as e:
                logger.error(f"Error closing database: {e!s}")

        except asyncio.CancelledError:
            logger.info("Shutdown cancelled - cleaning up remaining resources")
            try:
                # Ensure critical cleanup still happens
                if event_bus:
                    await event_bus.stop()
                if plugin_manager:
                    await plugin_manager.shutdown_plugins()
                # Only get database instance since it's a singleton
                db = await DatabaseManager.get_instance()
                await db.close()
            except Exception as e:
                logger.error(f"Error during emergency cleanup: {e!s}")
            finally:
                logger.info("Emergency cleanup complete")

        except Exception as e:
            logger.error(
                "Error during shutdown",
                extra={"error": str(e), "traceback": traceback.format_exc()},
            )
        finally:
            logger.info("Shutdown complete")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Recording Events API",
    description="API for handling start and end recording events",
    version="1.0.0",
    lifespan=lifespan,
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
directory_sync = None  # Add directory sync component

# API key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

# Track background tasks
background_tasks = weakref.WeakSet()


async def get_api_key(api_key_header: str = Depends(api_key_header)) -> str:
    """Validate API key from request header.

    Args:
        api_key_header: API key from request header

    Returns:
        Validated API key

    Raises:
        HTTPException: If API key is invalid
    """
    if api_key_header.lower() == os.getenv("API_KEY", "").lower():
        return api_key_header
    raise HTTPException(
        status_code=HTTP_UNAUTHORIZED,
        detail="Could not validate API key",
    )


@app.middleware("http")
async def api_logging_middleware(
    request: Request, call_next: Callable[[Request], Response]
) -> Response:
    """Middleware to log requests with detailed API endpoint information."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    start_time = datetime.now(UTC)

    try:
        response = await call_next(request)
        duration = (datetime.now(UTC) - start_time).total_seconds()

        logger.info(
            "API request completed",
            extra={
                "request_id": request_id,
                "response": {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "duration": duration,
                },
                "request": {
                    "method": request.method,
                    "url": str(request.url),
                    "headers": dict(request.headers),
                    "path_params": request.path_params,
                    "query_params": dict(request.query_params),
                },
            },
        )
        return response

    except asyncio.CancelledError:
        logger.info(
            "Request cancelled during shutdown",
            extra={
                "request_id": request_id,
                "request": {"method": request.method, "url": str(request.url)},
            },
        )
        # Don't re-raise CancelledError to allow graceful shutdown
        return Response(status_code=503, content="Service shutting down")

    except Exception as e:
        logger.error(
            "Error processing request",
            extra={
                "request_id": request_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "request": {"method": request.method, "url": str(request.url)},
            },
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
        },
    )
    return JSONResponse(
        status_code=HTTP_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body,
        },
    )


@app.post("/api/recording-started")
async def recording_started(
    background_tasks_fastapi: BackgroundTasks,
    request: RecordingStartRequest,
    x_api_key: str = Depends(get_api_key),
) -> dict[str, Any]:
    """Handle recording started event.

    Args:
        background_tasks_fastapi: FastAPI background tasks
        request: Recording start request model
        x_api_key: API key for authentication

    Returns:
        Response with recording ID and status
    """
    logger.info(
        "Processing recording started event",
        extra={"recording_id": request.recording_id},
    )

    # Create event
    event = request.to_event()

    # Return success immediately and process everything else in background
    background_tasks_fastapi.add_task(process_event, event)
    return {"status": "success", "recording_id": request.recording_id}


@app.post("/api/recording-ended")
async def recording_ended(
    background_tasks_fastapi: BackgroundTasks,
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
        },
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
        },
    )

    # Return success immediately and process everything else in background
    task = asyncio.create_task(process_event(event))
    background_tasks.add(task)
    task.add_done_callback(lambda t: background_tasks.discard(t))

    return {"status": "success", "event_id": event_id}


async def process_event(event: RecordingEvent) -> None:
    """Process recording event in background.

    Args:
        event: Recording event to process
    """
    try:
        # Create a new database connection for background processing
        db = await DatabaseManager.get_instance()

        # Check for duplicates
        if await event.is_duplicate():
            logger.info(
                "Skipping duplicate recording end event",
                extra={
                    "recording_id": event.recording_id,
                    "event_id": event.event_id,
                },
            )
            return

        # Create tasks for parallel execution
        save_task = asyncio.create_task(event.save())
        publish_task = asyncio.create_task(event_bus.publish(event))

        # Track tasks for cleanup
        background_tasks.add(save_task)
        background_tasks.add(publish_task)

        # Add callbacks for logging and cleanup
        def task_done_callback(task: asyncio.Task) -> None:
            try:
                background_tasks.discard(task)
                exc = task.exception()
                if exc:
                    logger.error(
                        "Task failed",
                        extra={
                            "recording_id": event.recording_id,
                            "event_id": event.event_id,
                            "error": str(exc),
                            "traceback": "".join(
                                traceback.format_exception(
                                    type(exc), exc, exc.__traceback__
                                )
                            ),
                        },
                    )
                else:
                    logger.debug(
                        "Task completed successfully",
                        extra={
                            "recording_id": event.recording_id,
                            "event_id": event.event_id,
                        },
                    )
            except asyncio.CancelledError:
                logger.debug(
                    "Task was cancelled",
                    extra={
                        "recording_id": event.recording_id,
                        "event_id": event.event_id,
                    },
                )
            except Exception as e:
                logger.error(
                    "Error in task callback",
                    extra={
                        "recording_id": event.recording_id,
                        "event_id": event.event_id,
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                )

        save_task.add_done_callback(task_done_callback)
        publish_task.add_done_callback(task_done_callback)

    except Exception as e:
        logger.error(
            "Error processing recording end event",
            extra={
                "recording_id": event.recording_id,
                "event_id": event.event_id,
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )


@app.get("/health")
async def health_check(api_key: str = Depends(get_api_key)):
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8001")),
        reload=True,
    )
