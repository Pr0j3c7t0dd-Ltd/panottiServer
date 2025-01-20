"""Tests for main FastAPI application."""

import asyncio
import os
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch
from unittest.mock import call
from concurrent.futures import Future
import types

import pytest
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from app.main import (
    app,
    api_logging_middleware,
    get_api_key,
    process_event,
    recording_ended,
    recording_started,
    shutdown,
    startup,
    validation_exception_handler,
)
from app.models.recording.events import RecordingEndRequest, RecordingStartRequest, RecordingEvent


@pytest.fixture(autouse=True)
def mock_threadpool():
    """Mock ThreadPoolExecutor to prevent shutdown issues."""
    with patch("concurrent.futures.ThreadPoolExecutor") as mock_executor:
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        yield mock_executor_instance


@pytest.fixture(autouse=True)
def override_dependencies():
    """Override dependencies for testing."""
    app.dependency_overrides[get_api_key] = lambda: "test_api_key"
    yield
    app.dependency_overrides = {}


@pytest.fixture
def test_client():
    """Create a test client."""
    # Create a new ThreadPool for each test
    with patch("app.models.database.ThreadPoolExecutor") as mock_threadpool, \
         patch("app.models.database.DatabaseManager.execute", new_callable=AsyncMock) as mock_execute:
        mock_threadpool_instance = mock_threadpool.return_value
        mock_threadpool_instance._shutdown = False  # Prevent shutdown issues
        
        # Create a real Future object for the submit method to return
        def submit_side_effect(*args, **kwargs):
            future = Future()
            future.set_result(None)  # Set a result to avoid blocking
            return future
            
        mock_threadpool_instance.submit.side_effect = submit_side_effect
        mock_execute.return_value = None  # Mock successful database connection
        
        with TestClient(app) as client:
            yield client


@pytest.fixture
def mock_background_tasks():
    """Mock background tasks."""
    return MagicMock()


@pytest.fixture
def mock_directory_sync():
    """Mock directory sync."""
    mock = MagicMock()
    mock.start_monitoring = MagicMock()
    mock.stop_monitoring = MagicMock()
    return mock


@pytest.fixture
async def mock_sleep():
    """Mock asyncio.sleep to return immediately."""
    with patch("asyncio.sleep", return_value=asyncio.Future()) as mock_sleep:
        mock_sleep.return_value.set_result(None)
        yield mock_sleep


@pytest.fixture
async def mock_event_bus(mock_sleep):
    """Mock event bus."""
    mock_bus = AsyncMock()
    mock_bus.start = AsyncMock()
    mock_bus.stop = AsyncMock()
    mock_bus.publish = AsyncMock()
    mock_bus._cleanup_task = None  # Ensure no background task is running
    
    # Create a dummy cleanup task that's already done
    mock_bus._cleanup_task = MagicMock()
    mock_bus._cleanup_task.done.return_value = True
    mock_bus._cleanup_task.cancel = MagicMock()
    
    yield mock_bus
    
    # Cleanup any tasks that might have been created
    if hasattr(mock_bus, '_cleanup_task') and mock_bus._cleanup_task is not None:
        mock_bus._cleanup_task.cancel()


@pytest.fixture
def mock_db_instance():
    """Mock database instance."""
    mock = AsyncMock()
    mock.execute = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def mock_db(mock_db_instance):
    """Mock database manager."""
    with patch("app.models.database.DatabaseManager") as mock_manager:
        mock_manager.get_instance = AsyncMock(return_value=mock_db_instance)
        mock_manager._instance = None
        yield mock_db_instance


@pytest.fixture
def mock_plugin_manager():
    """Mock plugin manager."""
    with patch("app.main.PluginManager") as mock_manager:
        mock_instance = AsyncMock()
        mock_manager.return_value = mock_instance
        mock_instance.discover_plugins = AsyncMock()
        mock_instance.initialize_plugins = AsyncMock()
        yield mock_manager


@pytest.fixture
def mock_threadpool():
    """Mock ThreadPoolExecutor."""
    with patch("app.models.database.ThreadPoolExecutor") as mock_pool:
        mock_pool_instance = mock_pool.return_value
        mock_pool_instance._shutdown = False
        
        # Create a real Future object for the submit method to return
        def submit_side_effect(*args, **kwargs):
            future = Future()
            future.set_result(None)  # Set a result to avoid blocking
            return future
            
        mock_pool_instance.submit.side_effect = submit_side_effect
        yield mock_pool


@pytest.mark.asyncio
async def test_get_api_key_invalid(test_client):
    """Test that invalid API keys are rejected."""
    app.dependency_overrides = {}  # Remove any overrides
    with patch.dict(os.environ, {"API_KEY": "test_api_key"}):
        response = test_client.get(
            "/health",
            headers={"X-API-Key": "invalid_key"}
        )
        assert response.status_code == 403
        assert response.json() == {"detail": "Could not validate API key"}
    app.dependency_overrides[get_api_key] = lambda: "test_api_key"  # Restore override for other tests


@pytest.mark.asyncio
async def test_get_api_key_valid(test_client):
    """Test that valid API keys are accepted."""
    with patch.dict(os.environ, {"API_KEY": "test_api_key"}):
        response = test_client.get(
            "/health",
            headers={"X-API-Key": "test_api_key"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_process_event(mock_event_bus, mock_db):
    """Test event processing."""
    mock_event = AsyncMock(spec=RecordingEvent)
    mock_event.recording_id = "test_id"
    mock_event.event_id = "test_event_id"
    mock_event.is_duplicate = AsyncMock(return_value=False)
    mock_event.save = AsyncMock()

    mock_tasks = set()
    mock_save_task = AsyncMock()
    mock_publish_task = AsyncMock()

    with patch("app.main.event_bus", mock_event_bus), \
         patch("app.main.background_tasks", mock_tasks), \
         patch("app.main.logger"), \
         patch("app.models.database.DatabaseManager.get_instance", return_value=mock_db), \
         patch("asyncio.create_task") as mock_create_task:

        # Configure create_task to return our mock tasks
        mock_create_task.side_effect = [mock_save_task, mock_publish_task]

        await process_event(mock_event)

        # Verify is_duplicate was called
        mock_event.is_duplicate.assert_awaited_once()
        assert not mock_event.is_duplicate.return_value

        # Verify tasks were created
        assert mock_create_task.call_count == 2
        assert mock_tasks == {mock_save_task, mock_publish_task}

        # Verify the callbacks were added
        mock_save_task.add_done_callback.assert_called_once()
        mock_publish_task.add_done_callback.assert_called_once()


def test_api_logging_middleware(test_client):
    """Test API logging middleware."""
    with patch("app.main.logger") as mock_logger:
        response = test_client.get("/health", headers={"X-API-Key": "test_api_key"})
        assert response.status_code in [200, 404]
        mock_logger.info.assert_called()


@pytest.mark.asyncio
async def test_validation_exception_handler():
    """Test validation exception handler."""
    # Create a mock request
    mock_request = MagicMock()
    mock_request.url.path = "/test"
    
    # Create a validation error
    exc = RequestValidationError(errors=[{"loc": ["body"], "msg": "field required", "type": "value_error.missing"}])
    
    # Call the handler
    response = await validation_exception_handler(mock_request, exc)
    
    # Verify response
    assert response.status_code == 422
    response_body = response.body.decode()
    assert "field required" in response_body
    assert "value_error.missing" in response_body


@pytest.mark.asyncio
async def test_recording_started_endpoint(test_client, mock_event_bus):
    """Test recording started endpoint."""
    with patch("app.main.event_bus", mock_event_bus):
        response = test_client.post(
            "/api/recording-started",
            headers={"X-API-Key": "test_api_key"},
            json={
                "meetingId": "test-meeting",
                "userId": "test-user",
                "timestamp": "2025-01-20T00:00:00Z",
                "recordingId": "test-recording",
                "filePath": "/path/to/recording.mp4"
            }
        )
        assert response.status_code == 200
        assert response.json() == {"status": "success", "recording_id": "test-recording"}


@pytest.mark.asyncio
async def test_recording_ended_endpoint(test_client, mock_event_bus):
    """Test recording ended endpoint."""
    with patch("app.main.event_bus", mock_event_bus):
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
                "metadata": {
                    "app": "test-app",
                    "version": "1.0.0"
                }
            }
        )
        assert response.status_code == 200
        response_json = response.json()
        assert response_json["status"] == "success"
        assert "event_id" in response_json
        assert response_json["event_id"].startswith("test-recording_ended_")