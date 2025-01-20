"""Tests for main FastAPI application."""

import asyncio
import os
from datetime import datetime, UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from fastapi.security.api_key import APIKeyHeader

from app.main import (
    get_api_key,
    process_event,
    validation_exception_handler,
    api_logging_middleware,
)
from app.models.recording.events import RecordingEndRequest, RecordingStartRequest, RecordingEvent


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
        raise HTTPException(
            status_code=403,
            detail="Could not validate API key"
        )
    
    @app.middleware("http")
    async def test_logging_middleware(request: Request, call_next):
        response = await call_next(request)
        logger = MagicMock()
        logger.info("API request completed")
        return response
    
    @app.get("/health")
    async def health(api_key: str = Depends(get_api_key_test)):
        return {"status": "ok"}
    
    @app.post("/api/recording-started")
    async def recording_started_test(request: RecordingStartRequest, api_key: str = Depends(get_api_key_test)):
        return {"status": "success", "recording_id": request.recording_id}
    
    @app.post("/api/recording-ended")
    async def recording_ended_test(request: RecordingEndRequest, api_key: str = Depends(get_api_key_test)):
        event_id = f"{request.recording_id}_ended_{datetime.now(UTC).timestamp()}"
        return {"status": "success", "event_id": event_id}
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client."""
    with TestClient(test_app) as client:
        yield client


def test_get_api_key_invalid(test_client):
    """Test that invalid API keys are rejected."""
    with patch.dict(os.environ, {"API_KEY": "test_api_key"}):
        response = test_client.get(
            "/health",
            headers={"X-API-Key": "invalid_key"}
        )
        assert response.status_code == 403
        assert response.json() == {"detail": "Could not validate API key"}


def test_get_api_key_valid(test_client):
    """Test that valid API keys are accepted."""
    with patch.dict(os.environ, {"API_KEY": "test_api_key"}):
        response = test_client.get(
            "/health",
            headers={"X-API-Key": "test_api_key"}
        )
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_process_event():
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
    exc = RequestValidationError(errors=[{"loc": ["body"], "msg": "field required", "type": "value_error.missing"}])
    response = await validation_exception_handler(mock_request, exc)
    assert response.status_code == 422
    response_body = response.body.decode()
    assert "field required" in response_body
    assert "value_error.missing" in response_body


def test_recording_started_endpoint(test_client):
    """Test recording started endpoint."""
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


def test_recording_ended_endpoint(test_client):
    """Test recording ended endpoint."""
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