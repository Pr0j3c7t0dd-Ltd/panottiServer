"""Tests for recording event handlers."""

from unittest.mock import AsyncMock, patch

import pytest

from app.core.events.handlers.recording import (
    handle_recording_ended,
    handle_recording_started,
)
from app.models.recording.events import RecordingEvent


@pytest.fixture
def mock_event_data():
    """Create a mock event data fixture."""
    return {
        "recording_timestamp": "2024-01-18T12:00:00",
        "recording_id": "test-recording-123",
        "system_audio_path": "/path/to/system.wav",
        "microphone_audio_path": "/path/to/mic.wav",
        "event": "recording.started",
        "metadata": {"event_title": "Test Meeting", "event_provider": "Test Provider"},
    }


@pytest.fixture
def mock_recording_event(mock_event_data):
    """Create a mock RecordingEvent fixture."""
    return RecordingEvent(**mock_event_data)


@pytest.mark.asyncio
async def test_handle_recording_started_with_dict(mock_event_data):
    """Test handling recording.started event with dictionary data."""
    with patch("app.core.events.handlers.recording.logger") as mock_logger:
        with patch.object(RecordingEvent, "save", new_callable=AsyncMock) as mock_save:
            await handle_recording_started(mock_event_data)

            mock_save.assert_called_once()
            mock_logger.info.assert_called_once_with(
                "Recording started",
                extra={
                    "recording_id": mock_event_data["recording_id"],
                    "event_id": mock_logger.info.call_args[1]["extra"]["event_id"],
                    "system_audio": mock_event_data["system_audio_path"],
                    "microphone_audio": mock_event_data["microphone_audio_path"],
                },
            )


@pytest.mark.asyncio
async def test_handle_recording_started_with_event(mock_recording_event):
    """Test handling recording.started event with RecordingEvent instance."""
    with patch("app.core.events.handlers.recording.logger") as mock_logger:
        with patch.object(RecordingEvent, "save", new_callable=AsyncMock) as mock_save:
            await handle_recording_started(mock_recording_event)

            mock_save.assert_called_once()
            mock_logger.info.assert_called_once_with(
                "Recording started",
                extra={
                    "recording_id": mock_recording_event.recording_id,
                    "event_id": mock_logger.info.call_args[1]["extra"]["event_id"],
                    "system_audio": mock_recording_event.system_audio_path,
                    "microphone_audio": mock_recording_event.microphone_audio_path,
                },
            )


@pytest.mark.asyncio
async def test_handle_recording_started_error():
    """Test handling recording.started event with error."""
    mock_data = {"invalid": "data"}

    with patch("app.core.events.handlers.recording.logger") as mock_logger:
        with pytest.raises(Exception):
            await handle_recording_started(mock_data)

            mock_logger.error.assert_called_once()
            assert (
                "Error handling recording.started event"
                in mock_logger.error.call_args[0][0]
            )


@pytest.mark.asyncio
async def test_handle_recording_ended_with_dict(mock_event_data):
    """Test handling recording.ended event with dictionary data."""
    mock_event_data["event"] = "recording.ended"

    with patch("app.core.events.handlers.recording.logger") as mock_logger:
        with patch.object(RecordingEvent, "save", new_callable=AsyncMock) as mock_save:
            await handle_recording_ended(mock_event_data)

            mock_save.assert_called_once()
            mock_logger.info.assert_called_once_with(
                "Recording ended",
                extra={
                    "recording_id": mock_event_data["recording_id"],
                    "event_id": mock_logger.info.call_args[1]["extra"]["event_id"],
                    "system_audio": mock_event_data["system_audio_path"],
                    "microphone_audio": mock_event_data["microphone_audio_path"],
                },
            )


@pytest.mark.asyncio
async def test_handle_recording_ended_with_event(mock_recording_event):
    """Test handling recording.ended event with RecordingEvent instance."""
    mock_recording_event.event = "recording.ended"

    with patch("app.core.events.handlers.recording.logger") as mock_logger:
        with patch.object(RecordingEvent, "save", new_callable=AsyncMock) as mock_save:
            await handle_recording_ended(mock_recording_event)

            mock_save.assert_called_once()
            mock_logger.info.assert_called_once_with(
                "Recording ended",
                extra={
                    "recording_id": mock_recording_event.recording_id,
                    "event_id": mock_logger.info.call_args[1]["extra"]["event_id"],
                    "system_audio": mock_recording_event.system_audio_path,
                    "microphone_audio": mock_recording_event.microphone_audio_path,
                },
            )


@pytest.mark.asyncio
async def test_handle_recording_ended_error():
    """Test handling recording.ended event with error."""
    mock_data = {"invalid": "data"}  # Missing required fields

    with patch("app.core.events.handlers.recording.logger") as mock_logger:
        with pytest.raises(Exception) as exc_info:
            await handle_recording_ended(mock_data)

        assert "validation error" in str(exc_info.value).lower()
        mock_logger.error.assert_called_once()
        assert "Error handling recording.ended event" in mock_logger.error.call_args[0][0]
