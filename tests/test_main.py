"""Tests for main.py FastAPI application."""

import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from app.main import app
from app.models.database import DatabaseManager
from app.models.recording.events import RecordingStartRequest, RecordingEndRequest
from app.plugins.events.bus import EventBus

pytestmark = pytest.mark.asyncio

@pytest.fixture
def mock_request():
    """Create a mock request with API key header."""
    request = MagicMock()
    request.headers = {"X-API-Key": "test_api_key"}
    return request

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    # Create mock instances
    mock_db_instance = AsyncMock()
    mock_event_bus_instance = AsyncMock()
    mock_plugin_manager_instance = AsyncMock()
    
    # Setup mock methods
    mock_event_bus_instance.start = AsyncMock()
    mock_event_bus_instance.stop = AsyncMock()
    mock_event_bus_instance.publish = AsyncMock()
    
    # Setup DatabaseManager singleton
    DatabaseManager._instance = mock_db_instance
    
    with patch('app.plugins.events.bus.EventBus', return_value=mock_event_bus_instance) as mock_event_bus, \
         patch('app.plugins.manager.PluginManager') as mock_plugin_manager, \
         patch('app.models.recording.events.RecordingEvent.is_duplicate', return_value=False):
        
        # Setup mocks
        mock_plugin_manager.get_instance.return_value = mock_plugin_manager_instance
        
        with TestClient(app) as client:
            yield client
            
    # Cleanup
    DatabaseManager._instance = None

def test_api_key_validation(test_client: TestClient) -> None:
    """Test API key validation."""
    response = test_client.get(
        "/api/recording-started",
        headers={"X-API-Key": "test_api_key"}
    )
    assert response.status_code != 403

def test_recording_started_endpoint(test_client: TestClient) -> None:
    """Test recording started endpoint."""
    recording_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    
    request_data = {
        "recordingId": recording_id,
        "recordingTimestamp": timestamp,
        "systemAudioPath": "/path/to/system.wav",
        "microphoneAudioPath": "/path/to/mic.wav",
        "metadata": {"test": "data"}
    }
    
    response = test_client.post(
        "/api/recording-started",
        headers={"X-API-Key": "test_api_key"},
        json=request_data
    )
    
    assert response.status_code == 200
    assert response.json() == {"status": "success", "recording_id": recording_id}

def test_recording_ended_endpoint(test_client: TestClient) -> None:
    """Test recording ended endpoint."""
    recording_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    
    request_data = {
        "recordingId": recording_id,
        "recordingTimestamp": timestamp,
        "systemAudioPath": "/path/to/system.wav",
        "microphoneAudioPath": "/path/to/mic.wav",
        "metadata": {"test": "data"}
    }
    
    response = test_client.post(
        "/api/recording-ended",
        headers={"X-API-Key": "test_api_key"},
        json=request_data
    )
    
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["status"] == "success"
    assert response_data["event_id"] == f"{recording_id}_ended_{timestamp}"

def test_validation_error_handling(test_client: TestClient) -> None:
    """Test validation error handling."""
    response = test_client.post(
        "/api/recording-started",
        headers={"X-API-Key": "test_api_key"},
        json={"invalid": "data"}
    )
    assert response.status_code == 422 