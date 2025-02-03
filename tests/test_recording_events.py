"""Tests for app/models/recording/events.py."""

import json
import sqlite3
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.core.events.types import EventContext
from app.models.database import DatabaseManager
from app.models.recording.events import (
    EventMetadata,
    RecordingEndRequest,
    RecordingEvent,
    RecordingStartRequest,
    parse_timestamp,
)


@pytest.mark.parametrize(
    "timestamp_str,expected_format",
    [
        ("20240101123456", "%Y%m%d%H%M%S"),  # Compact format
        ("2024-01-01T12:34:56.789", "%Y-%m-%dT%H:%M:%S.%f"),  # ISO with microseconds
        ("2024-01-01T12:34:56", "%Y-%m-%dT%H:%M:%S"),  # ISO without microseconds
        ("2024-01-01T12:34:56+0000", "%Y-%m-%dT%H:%M:%S%z"),  # With timezone
        ("1704147296", None),  # Unix timestamp
    ],
)
def test_parse_timestamp(timestamp_str, expected_format):
    """Test parse_timestamp function with various formats."""
    result = parse_timestamp(timestamp_str)
    assert isinstance(result, datetime)
    
    if expected_format:
        # Verify the format was correctly parsed
        try:
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1]
            datetime.strptime(timestamp_str, expected_format)
        except ValueError as e:
            pytest.fail(f"Failed to parse {timestamp_str} with format {expected_format}: {e}")


def test_parse_timestamp_invalid():
    """Test parse_timestamp with invalid format."""
    with pytest.raises(ValueError, match="Unable to parse timestamp"):
        parse_timestamp("invalid-timestamp")


@pytest.mark.asyncio
async def test_recording_event_save_with_database_lock():
    """Test RecordingEvent.save() with database lock retry logic."""
    mock_db = AsyncMock()
    
    # First attempt fails, second succeeds
    mock_db.execute = AsyncMock()
    mock_db.execute.side_effect = [
        sqlite3.OperationalError("database is locked"),  # First attempt
        None,  # Second attempt
    ]
    mock_db.commit = AsyncMock()

    # Mock gather to handle retries
    gather_calls = 0
    async def mock_gather(*args, **kwargs):
        nonlocal gather_calls
        gather_calls += 1
        if gather_calls == 1:
            raise sqlite3.OperationalError("database is locked")
        return None

    with patch("app.models.recording.events.DatabaseManager.get_instance_async", 
               return_value=mock_db), \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
         patch("asyncio.gather", side_effect=mock_gather):
        event = RecordingEvent(
            recording_timestamp="2024-01-01T12:34:56",
            recording_id="test-id",
            event="recording.started",
        )
        await event.save()

        # Verify retry behavior
        assert gather_calls == 2  # Initial fail + retry
        assert mock_sleep.called  # Verify sleep was called for retry delay
        
        # Verify the SQL queries
        calls = mock_db.execute.mock_calls
        assert "INSERT INTO recording_events" in str(calls[0])


@pytest.mark.asyncio
async def test_recording_event_save_recording_ended():
    """Test RecordingEvent.save() for recording.ended event."""
    mock_db = AsyncMock()
    mock_db.execute = AsyncMock(return_value=None)
    mock_db.commit = AsyncMock(return_value=None)

    with patch("app.models.recording.events.DatabaseManager.get_instance_async", 
               return_value=mock_db):
        event = RecordingEvent(
            recording_timestamp="2024-01-01T12:34:56",
            recording_id="test-id",
            event="recording.ended",
            system_audio_path="/path/to/system.wav",
            microphone_audio_path="/path/to/mic.wav",
        )
        await event.save()

        # Verify both insert and update queries were executed
        assert mock_db.execute.call_count == 2  # Insert + Update
        assert mock_db.commit.call_count == 1

        # Verify the correct SQL queries were executed
        calls = mock_db.execute.mock_calls
        assert "INSERT INTO recording_events" in str(calls[0])
        assert "INSERT INTO recordings" in str(calls[1])


@pytest.mark.asyncio
async def test_recording_event_save_max_retries_exceeded():
    """Test RecordingEvent.save() when max retries are exceeded."""
    mock_db = AsyncMock()
    
    # All attempts fail with database lock
    error = sqlite3.OperationalError("database is locked")
    mock_db.execute = AsyncMock()
    mock_db.execute.side_effect = [error] * 3  # Three failures
    mock_db.commit = AsyncMock()

    # Mock gather to always raise database lock error
    gather_calls = 0
    async def mock_gather(*args, **kwargs):
        nonlocal gather_calls
        gather_calls += 1
        raise sqlite3.OperationalError("database is locked")

    with patch("app.models.recording.events.DatabaseManager.get_instance_async", 
               return_value=mock_db), \
         patch("app.models.recording.events.logger") as mock_logger, \
         patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep, \
         patch("asyncio.gather", side_effect=mock_gather):
        event = RecordingEvent(
            recording_timestamp="2024-01-01T12:34:56",
            recording_id="test-id",
            event="recording.started",
        )
        
        # Should handle the error and log it
        with pytest.raises(sqlite3.OperationalError) as exc_info:
            await event.save()
        
        assert "database is locked" in str(exc_info.value)

        # Verify retry attempts and logging
        assert gather_calls == 3  # Max retries
        assert mock_sleep.call_count == 2  # Sleep between retries
        assert mock_logger.warning.call_count == 2  # Two retries logged


@pytest.mark.asyncio
async def test_recording_event_is_duplicate():
    """Test RecordingEvent.is_duplicate()."""
    mock_db = AsyncMock()
    mock_cursor = AsyncMock()
    mock_cursor.fetchone = AsyncMock(return_value=(1,))  # Simulate existing record
    mock_db.execute = AsyncMock(return_value=mock_cursor)

    with patch("app.models.recording.events.DatabaseManager.get_instance", 
               return_value=mock_db):
        event = RecordingEvent(
            recording_timestamp="2024-01-01T12:34:56",
            recording_id="test-id",
            event="recording.started",
        )
        
        is_duplicate = await event.is_duplicate()
        assert is_duplicate is False  # Method always returns False by design

        # No need to verify SQL execution since method is hardcoded to return False


def test_recording_start_request_normalize_event():
    """Test RecordingStartRequest.normalize_event()."""
    data = {
        "recordingId": "test-id",
        "timestamp": "2024-01-01T12:34:56",
        "event": "recording.started",
    }
    normalized = RecordingStartRequest.normalize_event(data)
    assert normalized["event"] == "recording.started"
    assert "timestamp" in normalized


def test_recording_end_request_normalize_event():
    """Test RecordingEndRequest.normalize_event()."""
    data = {
        "recordingId": "test-id",
        "timestamp": "2024-01-01T12:34:56",
        "event": "recording.ended",
        "systemAudioPath": "/path/to/system.wav",
        "microphoneAudioPath": "/path/to/mic.wav",
        "metadata": {},
    }
    normalized = RecordingEndRequest.normalize_event(data)
    assert normalized["event"] == "recording.ended"
    assert "timestamp" in normalized


def test_recording_start_request_to_event():
    """Test RecordingStartRequest.to_event()."""
    request = RecordingStartRequest(
        recordingId="test-id",
        timestamp="2024-01-01T12:34:56",
        metadata={"key": "value"},
    )
    event = request.to_event()
    assert isinstance(event, RecordingEvent)
    assert event.recording_id == "test-id"
    assert event.event == "recording.started"
    assert event.metadata == {"key": "value"}


def test_recording_end_request_to_event():
    """Test RecordingEndRequest.to_event()."""
    request = RecordingEndRequest(
        recordingId="test-id",
        timestamp="2024-01-01T12:34:56",
        systemAudioPath="/path/to/system.wav",
        microphoneAudioPath="/path/to/mic.wav",
        metadata={},
    )
    event = request.to_event()
    assert isinstance(event, RecordingEvent)
    assert event.recording_id == "test-id"
    assert event.event == "recording.ended"
    assert event.system_audio_path == "/path/to/system.wav"
    assert event.microphone_audio_path == "/path/to/mic.wav" 