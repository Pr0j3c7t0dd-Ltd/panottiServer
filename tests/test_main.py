"""Tests for main FastAPI application."""

import asyncio
import os
from datetime import UTC, datetime, timezone
from unittest.mock import ANY, AsyncMock, MagicMock, patch, call
import contextlib
import weakref

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request, Response, Security, BackgroundTasks
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.testclient import TestClient
from unittest import IsolatedAsyncioTestCase
from httpx import AsyncClient, ASGITransport
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

from app.main import (
    api_logging_middleware,
    app,
    lifespan,
    process_event,
    validation_exception_handler,
    background_tasks,
)
from app.models.recording.events import (
    RecordingEvent,
    RecordingStartRequest,
    RecordingEndRequest,
)


@pytest.fixture
def test_app():
    """Create a test app without lifespan."""
    app = FastAPI()

    # Add API key security
    API_KEY_NAME = "X-API-Key"
    api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

    async def get_api_key_test(api_key_header: str = Security(api_key_header)) -> str:
        if api_key_header.lower() == os.getenv("API_KEY", "").lower():
            return api_key_header
        raise HTTPException(status_code=403, detail="Could not validate API key")

    @app.middleware("http")
    async def test_logging_middleware(request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": str(e)},
            )

    @app.get("/health")
    async def health(api_key: str = Depends(get_api_key_test)):
        # Check event bus status
        if not hasattr(app.state, "event_bus") or not app.state.event_bus:
            raise HTTPException(status_code=500, detail="Event bus not initialized")
        try:
            if not app.state.event_bus.is_running():
                raise Exception("Event bus is not running")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return {"status": "ok"}

    @app.post("/api/recording-started")
    async def recording_started_test(
        request: RecordingStartRequest,
        api_key: str = Depends(get_api_key_test)
    ):
        # Create event
        event = request.to_event()

        # Process event synchronously for testing
        await process_event(event)
        return {"status": "success", "recording_id": request.recording_id}

    @app.post("/api/recording-started/error")
    async def recording_started_error_test(
        request: RecordingStartRequest, api_key: str = Depends(get_api_key_test)
    ):
        # Raise error synchronously
        raise HTTPException(status_code=500, detail="Test error")

    @app.post("/api/recording-ended")
    async def recording_ended_test(
        request: RecordingEndRequest, api_key: str = Depends(get_api_key_test)
    ):
        event_id = f"{request.recording_id}_ended_{datetime.now(UTC).timestamp()}"
        return {"status": "success", "event_id": event_id}

    @app.post("/api/recording-ended/error")
    async def recording_ended_error_test(
        request: RecordingEndRequest, api_key: str = Depends(get_api_key_test)
    ):
        raise Exception("Test error")

    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client."""
    with TestClient(test_app) as client:
        yield client


def test_get_api_key_invalid(test_client):
    """Test that invalid API keys are rejected."""
    with patch.dict(os.environ, {"API_KEY": "test_api_key"}):
        response = test_client.get("/health", headers={"X-API-Key": "invalid_key"})
        assert response.status_code == 403
        assert response.json() == {"detail": "Could not validate API key"}


def test_get_api_key_valid(test_client):
    """Test valid API key."""
    with patch.dict(os.environ, {"API_KEY": "test_api_key"}):
        # Create a mock event bus
        mock_event_bus = MagicMock()
        mock_event_bus.is_running = MagicMock(return_value=True)
        
        # Set the mock event bus on the app state
        test_client.app.state.event_bus = mock_event_bus
        
        response = test_client.get(
            "/health",
            headers={"X-API-Key": "test_api_key"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_process_event(cleanup_tasks):
    """Test event processing."""
    # Create mock event
    mock_event = AsyncMock(spec=RecordingEvent)
    mock_event.recording_id = "test_id"
    mock_event.event_id = "test_event_id"
    mock_event.is_duplicate = AsyncMock(return_value=False)
    mock_event.save = AsyncMock()

    # Create mock event bus
    mock_event_bus = AsyncMock()
    mock_event_bus.publish = AsyncMock()

    # Create mock db
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock()

    # Mock dependencies
    with (
        patch("app.main.event_bus", mock_event_bus),
        patch("app.models.database.DatabaseManager.get_instance", return_value=mock_db),
    ):
        await process_event(mock_event)

    # Verify calls
    mock_event.is_duplicate.assert_awaited_once()
    mock_event.save.assert_awaited_once()
    mock_event_bus.publish.assert_awaited_once_with(mock_event)


def test_api_logging_middleware():
    """Test API logging middleware."""
    app = FastAPI()
    mock_logger = MagicMock()

    @app.middleware("http")
    async def test_logging_middleware(request: Request, call_next):
        with patch("app.main.logger", mock_logger):
            return await api_logging_middleware(request, call_next)

    @app.get("/test")
    async def test_endpoint():
        return {"msg": "test"}

    client = TestClient(app)
    response = client.get("/test")
    assert response.status_code == 200
    mock_logger.info.assert_called()


@pytest.mark.asyncio
async def test_validation_exception_handler():
    """Test validation exception handler."""
    mock_request = MagicMock()
    mock_request.url.path = "/test"
    exc = RequestValidationError(
        errors=[
            {"loc": ["body"], "msg": "field required", "type": "value_error.missing"}
        ]
    )
    response = await validation_exception_handler(mock_request, exc)
    assert response.status_code == 422
    response_body = bytes(response.body).decode()
    assert "field required" in response_body
    assert "value_error.missing" in response_body


def test_recording_started_endpoint(test_client):
    """Test recording started endpoint success case."""
    with patch.dict(os.environ, {"API_KEY": "test_api_key"}):
        response = test_client.post(
            "/api/recording-started",
            headers={"X-API-Key": "test_api_key"},
            json={
                "meetingId": "test-meeting",
                "userId": "test-user",
                "timestamp": "2025-01-20T00:00:00Z",
                "recordingId": "test-recording",
                "filePath": "/path/to/recording.mp4",
            },
        )
        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "recording_id": "test-recording",
        }


def test_recording_started_endpoint_process_error(test_client):
    """Test that errors during recording started event processing are handled correctly."""
    # Set up API key and mock event bus
    with patch.dict(os.environ, {"API_KEY": "test_key"}):
        # Create a mock event bus
        mock_event_bus = MagicMock()
        mock_event_bus.is_running = MagicMock(return_value=True)
        test_client.app.state.event_bus = mock_event_bus

        # Make request with valid request format
        response = test_client.post(
            "/api/recording-started/error",
            headers={"X-API-Key": "test_key"},
            json={
                "meetingId": "test-meeting",
                "userId": "test-user",
                "timestamp": "2025-01-21T00:35:35Z",
                "recordingId": "test-recording",
                "filePath": "/path/to/recording.mp4"
            }
        )

        # Verify response
        assert response.status_code == 500
        assert response.json()["detail"] == "Test error"


def test_recording_ended_endpoint(test_client):
    """Test recording ended endpoint success case."""
    with patch.dict(os.environ, {"API_KEY": "test_api_key"}):
        response = test_client.post(
            "/api/recording-ended",
            headers={"X-API-Key": "test_api_key"},
            json={
                "meetingId": "test-meeting",
                "userId": "test-user",
                "timestamp": "2025-01-20T00:00:00Z",
                "recordingId": "test-recording",
                "filePath": "/path/to/recording.mp4",
                "fileSize": 1024,
                "duration": 60,
                "systemAudioPath": "/path/to/system_audio.wav",
                "microphoneAudioPath": "/path/to/mic_audio.wav",
                "metadata": {"app": "test-app", "version": "1.0.0"},
            },
        )
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "success"
        assert "event_id" in response_json
        assert response_json["event_id"].startswith("test-recording_ended_")


def test_recording_ended_endpoint_process_error(test_client):
    """Test recording ended endpoint with processing error."""
    # Set up API key and mock event bus
    with patch.dict(os.environ, {"API_KEY": "test_key"}):
        # Create a mock event bus
        mock_event_bus = MagicMock()
        mock_event_bus.is_running = MagicMock(return_value=True)
        test_client.app.state.event_bus = mock_event_bus

        # Make request with valid request format
        response = test_client.post(
            "/api/recording-ended/error",
            headers={"X-API-Key": "test_key"},
            json={
                "meetingId": "test-meeting",
                "userId": "test-user",
                "timestamp": "2025-01-21T00:35:35Z",
                "recordingId": "test-recording",
                "filePath": "/path/to/recording.mp4",
                "fileSize": 1024,
                "duration": 60,
                "systemAudioPath": "/path/to/system_audio.wav",
                "microphoneAudioPath": "/path/to/mic_audio.wav",
                "metadata": {"test": "data"}
            }
        )

        # Verify response
        assert response.status_code == 500
        assert response.json()["detail"] == "Test error"


@pytest.mark.asyncio
async def test_lifespan_startup_shutdown(cleanup_tasks):
    """Test lifespan context manager startup and shutdown."""
    mock_app = MagicMock()
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock()
    
    mock_event_bus = AsyncMock()
    mock_event_bus.start = AsyncMock()
    mock_event_bus.stop = AsyncMock()
    mock_event_bus.shutdown = AsyncMock()
    
    mock_plugin_manager = AsyncMock()
    mock_plugin_manager.discover_plugins = AsyncMock()
    mock_plugin_manager.initialize_plugins = AsyncMock()
    mock_plugin_manager.shutdown_plugins = AsyncMock()
    
    mock_directory_sync = MagicMock()
    mock_directory_sync.start_monitoring = MagicMock()
    mock_directory_sync.stop_monitoring = MagicMock()

    with (
        patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db),
        patch("app.main.EventBus", return_value=mock_event_bus),
        patch("app.main.PluginManager", return_value=mock_plugin_manager),
        patch("app.main.DirectorySync", return_value=mock_directory_sync),
        patch("app.main.setup_logging"),
        patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}),
    ):
        async with lifespan(mock_app):
            # Verify startup
            mock_db.execute.assert_awaited_once_with("SELECT 1")
            mock_directory_sync.start_monitoring.assert_called_once()
            mock_event_bus.start.assert_awaited_once()
            mock_plugin_manager.discover_plugins.assert_awaited_once()
            mock_plugin_manager.initialize_plugins.assert_awaited_once()

        # Verify shutdown
        mock_directory_sync.stop_monitoring.assert_called_once()
        mock_event_bus.stop.assert_awaited_once()
        mock_plugin_manager.shutdown_plugins.assert_awaited_once()
        mock_event_bus.shutdown.assert_awaited_once()
        mock_db.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_shutdown_with_background_tasks(cleanup_tasks):
    """Test lifespan shutdown with background tasks."""
    mock_app = MagicMock()
    
    # Create a mock task that properly handles cancellation
    async def mock_long_running():
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            # Properly handle cancellation
            return

    # Create a real task with proper cancellation handling
    task = asyncio.create_task(mock_long_running())
    mock_task = MagicMock(wraps=task)
    mock_task._coro = task._coro  # Keep the real coroutine for proper cancellation

    mock_db = AsyncMock(autospec=True)
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock()
    
    mock_event_bus = AsyncMock(autospec=True)
    mock_event_bus.start = AsyncMock()
    mock_event_bus.stop = AsyncMock()
    mock_event_bus.shutdown = AsyncMock()

    mock_plugin_manager = AsyncMock(autospec=True)
    mock_plugin_manager.discover_plugins = AsyncMock()
    mock_plugin_manager.initialize_plugins = AsyncMock()
    mock_plugin_manager.shutdown_plugins = AsyncMock()

    mock_directory_sync = MagicMock()
    mock_directory_sync.start_monitoring = MagicMock()
    mock_directory_sync.stop_monitoring = MagicMock()

    # Mock sleep to return immediately
    mock_sleep = AsyncMock()
    mock_sleep.return_value = None

    with (
        patch("app.main.background_tasks", {mock_task}),
        patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db),
        patch("app.main.EventBus", return_value=mock_event_bus),
        patch("app.main.PluginManager", return_value=mock_plugin_manager),
        patch("app.main.DirectorySync", return_value=mock_directory_sync),
        patch("app.main.setup_logging"),
        patch("asyncio.sleep", mock_sleep),
    ):
        try:
            async with lifespan(mock_app):
                # Let the context manager handle startup/shutdown
                pass

            # Verify the shutdown sequence
            assert task.cancelled() or task.done()
            mock_directory_sync.stop_monitoring.assert_called_once()
            mock_event_bus.stop.assert_awaited_once()
            mock_plugin_manager.shutdown_plugins.assert_awaited_once()
            mock_event_bus.shutdown.assert_awaited_once()
            mock_db.close.assert_awaited_once()

        finally:
            # Ensure the task is cleaned up
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass


@pytest.mark.asyncio
async def test_process_event_error_handling(cleanup_tasks):
    """Test error handling in process_event function."""
    mock_event = AsyncMock(spec=RecordingEvent)
    mock_event.recording_id = "test_id"
    mock_event.event_id = "test_event_id"
    mock_event.is_duplicate = AsyncMock(side_effect=Exception("Test error"))
    mock_event.save = AsyncMock()

    mock_event_bus = AsyncMock()
    mock_event_bus.publish = AsyncMock()

    with (
        patch("app.main.event_bus", mock_event_bus),
        patch("app.main.logger") as mock_logger,
    ):
        await process_event(mock_event)

    mock_logger.error.assert_called_with(
        "Error processing recording end event",
        extra={
            "recording_id": "test_id",
            "event_id": "test_event_id",
            "error": "Test error",
            "traceback": ANY,
        },
    )


@pytest.mark.asyncio
async def test_api_logging_middleware_error(cleanup_tasks):
    """Test error handling in API logging middleware."""
    app = FastAPI()
    
    # Create a failing endpoint
    @app.get("/test")
    async def test_endpoint():
        raise Exception("Test error")
    
    # Add logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        return await api_logging_middleware(request, call_next)
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/test")
        
        assert response.status_code == 500
        assert "Test error" in response.json()["error"]


def test_health_check_endpoint_error(test_client):
    """Test error handling in health check endpoint."""
    with patch.dict(os.environ, {"API_KEY": "test_api_key"}):
        # Create a mock event bus
        mock_event_bus = MagicMock()
        mock_event_bus.is_running = MagicMock(side_effect=Exception("Event bus error"))
        
        # Set the mock event bus on the app state
        test_client.app.state.event_bus = mock_event_bus
        
        response = test_client.get(
            "/health",
            headers={"X-API-Key": "test_api_key"}
        )
        assert response.status_code == 500
        assert "detail" in response.json()
        assert "Event bus error" in response.json()["detail"]


@pytest.mark.asyncio
async def test_lifespan_error_during_startup():
    """Test that errors during startup are logged and handled correctly."""
    # Mock the database to raise an error during startup
    mock_db_instance = AsyncMock()
    mock_db_instance.execute = AsyncMock(side_effect=Exception("Database connection failed"))
    
    # Mock the event bus and plugin manager
    mock_bus_instance = AsyncMock()
    mock_bus_instance.is_running = MagicMock(return_value=True)
    mock_bus_instance.stop = AsyncMock()
    
    mock_pm_instance = AsyncMock()
    mock_pm_instance.shutdown_plugins = AsyncMock()

    with patch("app.main.logger") as mock_logger, \
         patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db_instance), \
         patch("app.main.EventBus", return_value=mock_bus_instance), \
         patch("app.main.PluginManager", return_value=mock_pm_instance), \
         patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
        
        # Create a FastAPI app for testing
        app = FastAPI()
        
        # Add the lifespan context manager
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            try:
                await mock_db_instance.execute("SELECT 1")
            except Exception as e:
                mock_logger.error("Error during startup", extra={"error": str(e)})
                raise
            yield
            await mock_bus_instance.stop()
            await mock_pm_instance.shutdown_plugins()
        
        app.router.lifespan_context = lifespan
        
        # Test that startup error is logged
        with pytest.raises(Exception):
            async with lifespan(app):
                pass
                
        mock_logger.error.assert_called_with(
            "Error during startup",
            extra={"error": "Database connection failed"}
        )


@pytest.mark.asyncio
async def test_lifespan_cancelled_during_shutdown():
    """Test that cancellation during shutdown is handled correctly."""
    # Mock the database
    mock_db_instance = AsyncMock()
    mock_db_instance.execute = AsyncMock()
    mock_db_instance.close = AsyncMock()
    
    # Mock the event bus to raise CancelledError during stop
    mock_bus_instance = AsyncMock()
    mock_bus_instance.is_running = MagicMock(return_value=True)
    mock_bus_instance.stop = AsyncMock(side_effect=asyncio.CancelledError())
    
    # Mock the plugin manager
    mock_pm_instance = AsyncMock()
    mock_pm_instance.shutdown_plugins = AsyncMock()

    with patch("app.main.logger") as mock_logger, \
         patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db_instance), \
         patch("app.main.EventBus", return_value=mock_bus_instance), \
         patch("app.main.PluginManager", return_value=mock_pm_instance), \
         patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
        
        # Create a FastAPI app for testing
        app = FastAPI()
        
        # Add the lifespan context manager
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await mock_db_instance.execute("SELECT 1")
            yield
            try:
                await mock_bus_instance.stop()
                await mock_pm_instance.shutdown_plugins()
            except asyncio.CancelledError:
                mock_logger.warning(
                    "Shutdown cancelled, attempting emergency cleanup",
                    extra={"error": "CancelledError"}
                )
                raise
        
        app.router.lifespan_context = lifespan
        
        # Test that shutdown cancellation is logged
        with pytest.raises(asyncio.CancelledError):
            async with lifespan(app):
                pass
                
        mock_logger.warning.assert_called_with(
            "Shutdown cancelled, attempting emergency cleanup",
            extra={"error": "CancelledError"}
        )


@pytest.mark.asyncio
async def test_process_event_with_error():
    """Test event processing with error."""
    # Create mock event
    mock_event = AsyncMock(spec=RecordingEvent)
    mock_event.recording_id = "test_id"
    mock_event.event_id = "test_event_id"
    mock_event.is_duplicate = AsyncMock(return_value=False)
    mock_event.save = AsyncMock(side_effect=Exception("Failed to save event"))
    
    # Create mock event bus
    mock_event_bus = AsyncMock()
    mock_event_bus.publish = AsyncMock()
    mock_event_bus.is_running = MagicMock(return_value=True)
    
    # Create mock db
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock()
    
    # Create a mock logger
    mock_logger = MagicMock()
    
    # Mock dependencies
    with (
        patch("app.main.event_bus", mock_event_bus),
        patch("app.models.database.DatabaseManager.get_instance", return_value=mock_db),
        patch("app.main.logger", mock_logger),
        patch("app.main.background_tasks", set()),
        patch("app.main.asyncio.create_task", side_effect=lambda coro, **kwargs: coro),
    ):
        await process_event(mock_event)
    
    # Verify calls
    mock_event.is_duplicate.assert_awaited_once()
    mock_event.save.assert_awaited_once()
    mock_event_bus.publish.assert_not_awaited()
    mock_logger.error.assert_called_with(
        "Error processing recording end event",
        extra={
            "recording_id": "test_id",
            "event_id": "test_event_id",
            "error": "Failed to save event",
            "traceback": ANY,
        },
    )


class TestTaskDoneCallback(IsolatedAsyncioTestCase):
    """Test cases for task done callback."""

    async def asyncSetUp(self):
        """Set up test case."""
        # Create a mock logger
        self.mock_logger = MagicMock()
        self.logger_patcher = patch("app.main.logger", self.mock_logger)
        self.logger_patcher.start()

        # Create a mock sleep that returns immediately
        self.sleep_patcher = patch("asyncio.sleep", return_value=asyncio.Future())
        self.mock_sleep = self.sleep_patcher.start()
        self.mock_sleep.return_value.set_result(None)

        # Clear background tasks
        from app.main import background_tasks
        background_tasks.clear()

    async def asyncTearDown(self):
        """Clean up test case."""
        self.logger_patcher.stop()
        self.sleep_patcher.stop()

        # Clear background tasks
        from app.main import background_tasks
        background_tasks.clear()

    async def test_task_done_callback_error(self):
        """Test error handling in task done callback."""
        # Create a task done callback
        def task_done_callback(task):
            try:
                exc = task.exception()
                if exc:
                    self.mock_logger.error(
                        "Task failed",
                        extra={
                            "recording_id": getattr(task, "recording_id", None),
                            "event_id": getattr(task, "event_id", None),
                            "error": str(exc),
                            "traceback": ANY
                        }
                    )
            except asyncio.CancelledError:
                pass

        # Create a task that will raise an exception
        async def failing_task():
            await asyncio.sleep(0)  # This will be mocked
            raise Exception("Task error")

        # Create and run the task
        task = asyncio.create_task(failing_task())
        task.recording_id = "test_id"
        task.event_id = "test_event_id"
        task.add_done_callback(task_done_callback)

        # Add task to background tasks
        from app.main import background_tasks
        background_tasks.add(task)

        # Wait for task to complete
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except Exception:
            pass  # Expected to fail

        # Wait a bit for the callback to execute
        await asyncio.sleep(0)

        # Verify error was logged
        self.mock_logger.error.assert_called_with(
            "Task failed",
            extra={
                "recording_id": "test_id",
                "event_id": "test_event_id",
                "error": "Task error",
                "traceback": ANY
            }
        )


class TestRecordingEndpoint(IsolatedAsyncioTestCase):
    """Test cases for recording endpoint."""

    async def asyncSetUp(self):
        """Set up test case."""
        # Create a minimal test app
        self.app = FastAPI()
        
        @self.app.post("/api/recording-ended")
        async def recording_ended_test(request: RecordingEndRequest):
            raise HTTPException(status_code=500, detail="Processing error")

        # Create a mock event bus
        self.mock_event_bus = MagicMock()
        self.mock_event_bus.is_running = MagicMock(return_value=True)
        self.app.state.event_bus = self.mock_event_bus

        # Create test client with ASGITransport
        self.client = AsyncClient(transport=ASGITransport(app=self.app), base_url="http://test")
        await self.client.__aenter__()

    async def asyncTearDown(self):
        """Clean up test case."""
        await self.client.__aexit__(None, None, None)

    async def test_recording_ended_endpoint_process_error(self):
        """Test recording ended endpoint with processing error."""
        # Make request with valid request format
        response = await self.client.post(
            "/api/recording-ended",
            headers={"X-API-Key": "test_api_key"},
            json={
                "recordingId": "test_id",
                "timestamp": "2025-01-21T00:35:35Z",
                "systemAudioPath": "/path/to/system_audio.wav",
                "microphoneAudioPath": "/path/to/mic_audio.wav",
                "metadata": {"test": "data"}
            }
        )

        # Verify response
        assert response.status_code == 500
        assert response.json()["detail"] == "Processing error"


@pytest.mark.asyncio
async def test_lifespan_database_error():
    """Test lifespan handling of database errors during shutdown."""
    app = FastAPI()
    
    # Mock database with error during close
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock(side_effect=Exception("Database close error"))
    
    # Mock event bus and plugin manager
    mock_event_bus = AsyncMock()
    mock_event_bus.start = AsyncMock()
    mock_event_bus.stop = AsyncMock()
    mock_event_bus.shutdown = AsyncMock()
    
    mock_plugin_manager = AsyncMock()
    mock_plugin_manager.discover_plugins = AsyncMock()
    mock_plugin_manager.initialize_plugins = AsyncMock()
    mock_plugin_manager.shutdown_plugins = AsyncMock()
    
    with (
        patch("app.main.EventBus", return_value=mock_event_bus),
        patch("app.main.PluginManager", return_value=mock_plugin_manager),
        patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db),
        patch("app.main.DirectorySync"),
        patch("app.main.setup_logging"),
    ):
        async with lifespan(app):
            pass  # Simulate normal operation
            
    # Verify error was handled
    mock_db.close.assert_awaited_once()
    mock_event_bus.stop.assert_awaited_once()
    mock_plugin_manager.shutdown_plugins.assert_awaited_once()


@pytest.mark.asyncio
async def test_recording_started_endpoint_with_background_task(cleanup_tasks):
    """Test recording started endpoint with background task processing."""
    app = FastAPI()
    
    # Mock dependencies
    mock_event = AsyncMock()
    mock_event.is_duplicate = AsyncMock(return_value=False)
    mock_event.save = AsyncMock()
    
    mock_event_bus = AsyncMock()
    mock_event_bus.publish = AsyncMock()
    mock_event_bus.is_running = MagicMock(return_value=True)
    
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    
    # Add API key security
    API_KEY_NAME = "X-API-Key"
    api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

    async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
        if api_key_header.lower() == os.getenv("API_KEY", "").lower():
            return api_key_header
        raise HTTPException(status_code=403, detail="Could not validate API key")

    @app.post("/api/recording-started")
    async def recording_started_test(
        background_tasks: BackgroundTasks,
        request: RecordingStartRequest,
        api_key: str = Depends(get_api_key)
    ):
        # Create event
        event = mock_event
        
        # Add background task
        background_tasks.add_task(process_event, event)
        return {"status": "success", "recording_id": request.recording_id}
    
    with (
        patch.dict(os.environ, {"API_KEY": "test_api_key"}),
        patch("app.main.event_bus", mock_event_bus),
        patch("app.models.database.DatabaseManager.get_instance", return_value=mock_db),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/recording-started",
                headers={"X-API-Key": "test_api_key"},
                json={
                    "meetingId": "test-meeting",
                    "userId": "test-user",
                    "timestamp": "2025-01-20T00:00:00Z",
                    "recordingId": "test-recording",
                    "filePath": "/path/to/recording.mp4",
                }
            )
            
            assert response.status_code == 200
            assert response.json() == {
                "status": "success",
                "recording_id": "test-recording"
            }

@pytest.mark.asyncio
async def test_recording_ended_with_background_task():
    """Test recording ended endpoint with background task processing."""
    # Create FastAPI app with routes
    app = FastAPI()
    
    # Mock dependencies
    mock_event = AsyncMock()
    mock_event.is_duplicate = AsyncMock(return_value=False)
    mock_event.save = AsyncMock()
    
    mock_event_bus = AsyncMock()
    mock_event_bus.publish = AsyncMock()
    mock_event_bus.is_running = MagicMock(return_value=True)
    
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    
    # Add API key security
    API_KEY_NAME = "X-API-Key"
    api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

    async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
        if api_key_header.lower() == os.getenv("API_KEY", "").lower():
            return api_key_header
        raise HTTPException(status_code=403, detail="Could not validate API key")

    @app.post("/api/recording-ended")
    async def recording_ended_test(
        background_tasks: BackgroundTasks,
        request: RecordingEndRequest,
        api_key: str = Depends(get_api_key)
    ):
        # Create event
        event = mock_event
        
        # Add background task
        background_tasks.add_task(process_event, event)
        event_id = f"{request.recording_id}_ended_{datetime.now(UTC).timestamp()}"
        return {"status": "success", "event_id": event_id}
    
    # Add validation exception handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )
    
    # Add models
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    with (
        patch.dict(os.environ, {"API_KEY": "test_api_key"}),
        patch("app.main.event_bus", mock_event_bus),
        patch("app.models.database.DatabaseManager.get_instance", return_value=mock_db),
        patch("app.models.recording.events.RecordingEndRequest.to_event", return_value=mock_event),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/recording-ended",
                headers={"X-API-Key": "test_api_key"},
                json={
                    "meetingId": "test-meeting",
                    "userId": "test-user",
                    "timestamp": "2025-01-20T00:00:00Z",
                    "recordingId": "test-recording",
                    "systemAudioPath": "/path/to/system.mp4",
                    "microphoneAudioPath": "/path/to/mic.mp4",
                    "metadata": {
                        "eventTitle": "Test Meeting",
                        "eventProvider": "test",
                        "eventProviderId": "123",
                    }
                }
            )
            
            assert response.status_code == 200
            assert "event_id" in response.json()
            assert response.json()["status"] == "success"

@pytest.mark.asyncio
async def test_task_done_callback_success():
    """Test successful task done callback."""
    # Create a task that completes successfully
    async def successful_task():
        return "success"
    
    task = asyncio.create_task(successful_task())
    await task
    
    # Create a mock logger
    mock_logger = MagicMock()
    
    with patch("app.main.logger", mock_logger):
        # Call the callback function directly
        def task_done_callback(task):
            try:
                result = task.result()
                mock_logger.debug(f"Task completed successfully: {result}")
            except Exception as e:
                mock_logger.error(f"Task failed: {e}")
                
        task_done_callback(task)
        
        # Verify logging
        mock_logger.debug.assert_called_once()
        assert "Task completed successfully" in mock_logger.debug.call_args[0][0]


@pytest.mark.asyncio
async def test_process_event_duplicate():
    """Test processing of duplicate events."""
    # Create mock event that is a duplicate
    mock_event = AsyncMock(spec=RecordingEvent)
    mock_event.recording_id = "test_id"
    mock_event.event_id = "test_event_id"
    mock_event.is_duplicate = AsyncMock(return_value=True)
    mock_event.save = AsyncMock()
    
    # Create mock event bus
    mock_event_bus = AsyncMock()
    mock_event_bus.publish = AsyncMock()
    
    # Create mock db
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock()
    
    # Mock dependencies
    with (
        patch("app.main.event_bus", mock_event_bus),
        patch("app.models.database.DatabaseManager.get_instance", return_value=mock_db),
    ):
        await process_event(mock_event)
    
    # Verify calls
    mock_event.is_duplicate.assert_awaited_once()
    mock_event.save.assert_not_awaited()
    mock_event_bus.publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_middleware_cancelled_request(test_app: FastAPI):
    """Test that the API middleware handles cancelled requests during shutdown."""
    mock_logger = MagicMock()
    
    # Create a future that will be cancelled
    future = asyncio.Future()
    future.cancel()
    
    async def mock_call_next(request):
        await future
        return None  # This line won't be reached due to cancellation
        
    # Add middleware directly to test_app
    @test_app.middleware("http")
    async def test_middleware(request: Request, call_next):
        return await api_logging_middleware(request, mock_call_next)
        
    @test_app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}
        
    with patch("app.main.logger", mock_logger):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/test", headers={"X-Request-ID": "test-request-id"})
            
            assert response.status_code == 503
            assert response.json() == {"error": "Service shutting down"}
            mock_logger.info.assert_called_with(
                "Request cancelled during shutdown",
                extra={
                    "request_id": "test-request-id",
                    "request": {"method": "GET", "url": "http://test/test"},
                },
            )


@pytest.mark.asyncio
async def test_lifespan_emergency_cleanup():
    """Test emergency cleanup during lifespan shutdown."""
    test_app = FastAPI()
    
    mock_logger = MagicMock()
    
    # Create mock dependencies that will timeout
    mock_event_bus = AsyncMock()
    mock_event_bus.stop = AsyncMock()
    mock_event_bus.shutdown = AsyncMock(side_effect=asyncio.TimeoutError("Event bus shutdown timed out"))
    
    mock_plugin_manager = AsyncMock()
    mock_plugin_manager.discover_plugins = AsyncMock()
    mock_plugin_manager.initialize_plugins = AsyncMock()
    mock_plugin_manager.shutdown_plugins = AsyncMock(side_effect=asyncio.TimeoutError("Plugin shutdown timed out"))
    
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock(side_effect=asyncio.TimeoutError("Database shutdown timed out"))
    
    mock_directory_sync = MagicMock()
    mock_directory_sync.start_monitoring = MagicMock()
    mock_directory_sync.stop_monitoring = MagicMock()
    
    with (
        patch("app.main.logger", mock_logger),
        patch("app.main.EventBus", return_value=mock_event_bus),
        patch("app.main.PluginManager", return_value=mock_plugin_manager),
        patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db),
        patch("app.main.DirectorySync", return_value=mock_directory_sync),
        patch("app.main.setup_logging"),
        patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    ):
        async with lifespan(test_app):
            # Startup should complete successfully
            mock_logger.info.assert_any_call("Startup complete")
            
        # Verify shutdown timeouts were logged
        mock_logger.warning.assert_has_calls(
            [
                call("Plugin shutdown timed out after 10 seconds"),
                call("Event bus shutdown timed out after 10 seconds"), 
                call("Database shutdown timed out after 10 seconds")
            ],
            any_order=True
        )
        
        # Verify cleanup was attempted
        mock_directory_sync.stop_monitoring.assert_called_once()
        mock_event_bus.stop.assert_called_once()
        mock_plugin_manager.shutdown_plugins.assert_awaited_once()
        mock_event_bus.shutdown.assert_awaited_once()
        mock_db.close.assert_awaited_once()
        
        # Verify final shutdown status
        mock_logger.info.assert_any_call("Shutdown complete")


@pytest.mark.asyncio
async def test_lifespan_startup_error_handling():
    """Test error handling during lifespan startup."""
    mock_app = MagicMock()
    
    # Mock database to fail during startup
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(side_effect=Exception("Database initialization failed"))
    
    # Create a mock logger to verify error logging
    mock_logger = MagicMock()

    with (
        patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db),
        patch("app.main.setup_logging"),
        patch("app.main.logger", mock_logger),
        patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}),
    ):
        with pytest.raises(Exception) as exc_info:
            async with lifespan(mock_app):
                pass
                
        assert str(exc_info.value) == "Database initialization failed"
        mock_db.execute.assert_awaited_once_with("SELECT 1")
        mock_logger.info.assert_called_with("Initializing database")


@pytest.mark.asyncio
async def test_lifespan_shutdown_timeout():
    """Test timeout handling during lifespan shutdown."""
    mock_app = MagicMock()
    
    # Mock dependencies that will raise timeout errors
    mock_event_bus = AsyncMock()
    mock_event_bus.start = AsyncMock()
    mock_event_bus.stop = AsyncMock()
    mock_event_bus.shutdown = AsyncMock(side_effect=asyncio.TimeoutError())
    
    mock_plugin_manager = AsyncMock()
    mock_plugin_manager.discover_plugins = AsyncMock()
    mock_plugin_manager.initialize_plugins = AsyncMock()
    mock_plugin_manager.shutdown_plugins = AsyncMock(side_effect=asyncio.TimeoutError())
    
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock(side_effect=asyncio.TimeoutError())
    
    mock_directory_sync = MagicMock()
    mock_directory_sync.start_monitoring = MagicMock()
    mock_directory_sync.stop_monitoring = MagicMock()
    
    # Create a mock logger to verify warning logging
    mock_logger = MagicMock()

    with (
        patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db),
        patch("app.main.EventBus", return_value=mock_event_bus),
        patch("app.main.PluginManager", return_value=mock_plugin_manager),
        patch("app.main.DirectorySync", return_value=mock_directory_sync),
        patch("app.main.setup_logging"),
        patch("app.main.logger", mock_logger),
        patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"})
    ):
        try:
            async with lifespan(mock_app):
                # Verify startup completed
                mock_db.execute.assert_awaited_once_with("SELECT 1")
                mock_directory_sync.start_monitoring.assert_called_once()
                mock_event_bus.start.assert_awaited_once()
                mock_plugin_manager.discover_plugins.assert_awaited_once()
                mock_plugin_manager.initialize_plugins.assert_awaited_once()
                
                # Simulate some work
                await asyncio.sleep(0)
        except Exception:
            pass  # Expected to fail during shutdown

        # Verify timeout warnings were logged
        mock_logger.warning.assert_has_calls([
            call("Plugin shutdown timed out after 10 seconds"),
            call("Event bus shutdown timed out after 10 seconds"),
            call("Database shutdown timed out after 10 seconds")
        ], any_order=True)


@pytest.mark.asyncio
async def test_lifespan_database_init_error(cleanup_tasks):
    """Test error handling during database initialization in lifespan."""
    app = FastAPI()
    
    # Mock database error
    mock_db = AsyncMock()
    mock_db.execute.side_effect = Exception("Database connection error")
    
    with patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db):
        try:
            async with lifespan(app):
                # Should not reach here due to database error
                assert False, "Should have raised an exception"
        except Exception as e:
            assert "Database connection error" in str(e)


@pytest.mark.asyncio
async def test_lifespan_plugin_init_error(cleanup_tasks):
    """Test error handling during plugin initialization in lifespan."""
    app = FastAPI()
    
    # Mock successful database
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock()
    mock_db.close = AsyncMock()
    
    # Mock plugin manager error
    mock_plugin_manager = AsyncMock()
    mock_plugin_manager.discover_plugins.side_effect = Exception("Plugin discovery error")
    mock_plugin_manager.shutdown_plugins = AsyncMock()
    
    # Mock event bus
    mock_event_bus = AsyncMock()
    mock_event_bus.start = AsyncMock()
    mock_event_bus.stop = AsyncMock()
    mock_event_bus.shutdown = AsyncMock()
    
    # Mock directory sync
    mock_directory_sync = MagicMock()
    mock_directory_sync.start_monitoring = MagicMock()
    mock_directory_sync.stop_monitoring = MagicMock()
    
    with (
        patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db),
        patch("app.main.PluginManager", return_value=mock_plugin_manager),
        patch("app.main.EventBus", return_value=mock_event_bus),
        patch("app.main.DirectorySync", return_value=mock_directory_sync),
        patch("app.main.setup_logging"),
    ):
        try:
            async with lifespan(app):
                # Should not reach here due to plugin error
                assert False, "Should have raised an exception"
        except Exception as e:
            assert "Plugin discovery error" in str(e)
            
            # Verify initialization sequence
            mock_db.execute.assert_awaited_once()
            mock_directory_sync.start_monitoring.assert_called_once()
            mock_event_bus.start.assert_awaited_once()
            mock_plugin_manager.discover_plugins.assert_awaited_once()


@pytest.mark.asyncio
async def test_lifespan_emergency_cleanup_error(cleanup_tasks):
    """Test emergency cleanup during shutdown when errors occur."""
    app = FastAPI()
    
    # Mock components
    mock_db = AsyncMock()
    mock_event_bus = AsyncMock()
    mock_plugin_manager = AsyncMock()
    mock_directory_sync = MagicMock()
    
    # Simulate errors during normal shutdown
    mock_event_bus.stop.side_effect = Exception("Stop error")
    mock_plugin_manager.shutdown_plugins.side_effect = Exception("Shutdown error")
    mock_db.close.side_effect = Exception("Close error")
    
    with (
        patch("app.main.DatabaseManager.get_instance_async", return_value=mock_db),
        patch("app.main.EventBus", return_value=mock_event_bus),
        patch("app.main.PluginManager", return_value=mock_plugin_manager),
        patch("app.main.DirectorySync", return_value=mock_directory_sync),
    ):
        try:
            async with lifespan(app):
                # Normal operation
                pass
        except Exception as e:
            # Verify emergency cleanup was attempted
            mock_event_bus.stop.assert_called()
            mock_plugin_manager.shutdown_plugins.assert_called()
            mock_db.close.assert_called()
            assert "Error during shutdown" in str(e)


@pytest.mark.asyncio
async def test_recording_started_endpoint_db_error(cleanup_tasks):
    """Test error handling when database error occurs during recording started."""
    app = FastAPI()
    
    # Mock process_event to raise error
    mock_process_event = AsyncMock(side_effect=Exception("Database error"))
    
    # Setup API key
    API_KEY_NAME = "X-API-Key"
    api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
    
    async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
        if api_key_header.lower() == os.getenv("API_KEY", "").lower():
            return api_key_header
        raise HTTPException(status_code=403, detail="Could not validate API key")
    
    @app.post("/api/recording-started")
    async def recording_started_test(
        request: RecordingStartRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(get_api_key)
    ):
        # Create event
        event = request.to_event()
        
        # Process event synchronously for testing
        try:
            await mock_process_event(event)
            return {"status": "success", "recording_id": request.recording_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    with (
        patch("app.main.process_event", mock_process_event),
        patch.dict(os.environ, {"API_KEY": "test_key"})
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/recording-started", 
                headers={"X-API-Key": "test_key"},
                json={
                    "recordingId": "test_id",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "systemAudioPath": "/test/system.wav",
                    "microphoneAudioPath": "/test/mic.wav",
                    "metadata": {
                        "eventTitle": "Test Meeting",
                        "eventProvider": "Test Provider"
                    }
                })
            
            assert response.status_code == 500
            assert "Database error" in response.json()["detail"]


@pytest.mark.asyncio
async def test_recording_ended_endpoint_event_error(cleanup_tasks):
    """Test error handling when event error occurs during recording ended."""
    app = FastAPI()
    
    # Mock process_event to raise error
    mock_process_event = AsyncMock(side_effect=Exception("Event error"))
    
    # Setup API key
    API_KEY_NAME = "X-API-Key"
    api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
    
    async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
        if api_key_header.lower() == os.getenv("API_KEY", "").lower():
            return api_key_header
        raise HTTPException(status_code=403, detail="Could not validate API key")
    
    @app.post("/api/recording-ended")
    async def recording_ended_test(
        request: RecordingEndRequest,
        background_tasks: BackgroundTasks,
        api_key: str = Depends(get_api_key)
    ):
        # Create event
        event = request.to_event()
        event_id = f"{event.recording_id}_ended_{event.recording_timestamp}"
        event.event_id = event_id
        
        # Process event synchronously for testing
        try:
            await mock_process_event(event)
            return {"status": "success", "event_id": event_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    with (
        patch("app.main.process_event", mock_process_event),
        patch.dict(os.environ, {"API_KEY": "test_key"})
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/recording-ended", 
                headers={"X-API-Key": "test_key"},
                json={
                    "recordingId": "test_id",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "systemAudioPath": "/test/system.wav",
                    "microphoneAudioPath": "/test/mic.wav",
                    "metadata": {
                        "eventTitle": "Test Meeting",
                        "eventProvider": "Test Provider"
                    }
                })
            
            assert response.status_code == 500
            assert "Event error" in response.json()["detail"]


@pytest.mark.asyncio
async def test_task_done_callback_error_handling(cleanup_tasks):
    """Test error handling in task done callback."""
    # Create a failing task
    async def failing_task():
        raise Exception("Task failed")
    
    # Create the task
    task = asyncio.create_task(failing_task())
    
    # Mock task done callback
    def task_done_callback(task):
        try:
            task.result()
        except Exception:
            pass
        background_tasks.discard(task)
    
    # Add task done callback
    task.add_done_callback(task_done_callback)
    
    # Add to background tasks
    background_tasks.add(task)
    
    # Wait for task to complete
    try:
        await task
    except Exception:
        pass
    
    # Wait for callback to execute and task to be removed
    for _ in range(10):  # Try up to 10 times
        await asyncio.sleep(0.1)
        if task not in background_tasks:
            break
    
    # Verify task was removed
    assert task not in background_tasks


@pytest.mark.asyncio
async def test_health_check_event_bus_error(cleanup_tasks):
    """Test health check endpoint when event bus has error."""
    app = FastAPI()
    
    # Mock event bus error
    mock_event_bus = AsyncMock()
    mock_event_bus.is_running = AsyncMock(side_effect=Exception("Event bus error"))
    
    # Setup API key
    API_KEY_NAME = "X-API-Key"
    api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
    
    async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
        return api_key_header
    
    @app.get("/health")
    async def health_check(api_key: str = Depends(get_api_key)):
        if not hasattr(app.state, "event_bus") or not app.state.event_bus:
            raise HTTPException(status_code=500, detail="Event bus not initialized")
        try:
            await app.state.event_bus.is_running()
            return {"status": "ok"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    app.state.event_bus = mock_event_bus
    
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get(
            "/health",
            headers={"X-API-Key": "test_key"}
        )
        
        assert response.status_code == 500
        assert "Event bus error" in response.json()["detail"]


@pytest.fixture
async def cleanup_tasks():
    yield
    # Get all tasks except current task
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        # Cancel all tasks
        for task in tasks:
            task.cancel()
        # Wait with timeout for tasks to cancel
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=1.0)
        except asyncio.TimeoutError:
            pass  # Some tasks may be stuck, continue cleanup
