"""Tests for recording event handlers."""
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def mock_recording_event():
    """Create a mock recording event."""
    return {
        "recording_id": "test-recording-123",
        "event_id": "test-event-123",
        "recording_timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "recording.started",
        "system_audio_path": "/path/to/system/audio",
        "microphone_audio_path": "/path/to/mic/audio"
    }


@pytest.fixture
def mock_recording_event_obj():
    """Create a mock RecordingEvent object."""
    mock_obj = MagicMock()
    mock_obj.save = AsyncMock()
    mock_obj.recording_id = "test-recording-123"
    mock_obj.event_id = "test-event-123"
    mock_obj.system_audio_path = "/path/to/system/audio"
    mock_obj.microphone_audio_path = "/path/to/mic/audio"
    return mock_obj


@pytest.mark.asyncio
async def test_handle_recording_started_dict(mock_recording_event, mock_recording_event_obj):
    """Test handling recording.started event with dict input."""
    mock_recording_event_cls = type('RecordingEvent', (), {
        '__call__': lambda self, **kwargs: mock_recording_event_obj,
        '__instancecheck__': lambda self, obj: False
    })()
    
    with patch('app.models.recording.events.RecordingEvent', mock_recording_event_cls):
        from app.core.events.handlers.recording import handle_recording_started
        await handle_recording_started(mock_recording_event)
        mock_recording_event_obj.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_recording_started_event_obj(mock_recording_event_obj):
    """Test handling recording.started event with RecordingEvent input."""
    mock_recording_event_cls = type('RecordingEvent', (), {
        '__instancecheck__': lambda self, obj: isinstance(obj, MagicMock)
    })()
    
    with patch('app.models.recording.events.RecordingEvent', mock_recording_event_cls):
        from app.core.events.handlers.recording import handle_recording_started
        await handle_recording_started(mock_recording_event_obj)
        mock_recording_event_obj.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_recording_started_error(mock_recording_event):
    """Test handling recording.started event with error."""
    def raise_error(**kwargs):
        raise ValueError("Test error")
        
    mock_recording_event_cls = type('RecordingEvent', (), {
        '__call__': lambda self, **kwargs: raise_error(**kwargs),
        '__instancecheck__': lambda self, obj: False
    })()
    
    with patch('app.models.recording.events.RecordingEvent', mock_recording_event_cls):
        from app.core.events.handlers.recording import handle_recording_started
        with pytest.raises(ValueError):
            await handle_recording_started(mock_recording_event)


@pytest.mark.asyncio
async def test_handle_recording_ended_dict(mock_recording_event, mock_recording_event_obj):
    """Test handling recording.ended event with dict input."""
    # Update event type for ended event
    mock_recording_event["event"] = "recording.ended"
    mock_recording_event_cls = type('RecordingEvent', (), {
        '__call__': lambda self, **kwargs: mock_recording_event_obj,
        '__instancecheck__': lambda self, obj: False
    })()
    
    with patch('app.models.recording.events.RecordingEvent', mock_recording_event_cls):
        from app.core.events.handlers.recording import handle_recording_ended
        await handle_recording_ended(mock_recording_event)
        mock_recording_event_obj.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_recording_ended_event_obj(mock_recording_event_obj):
    """Test handling recording.ended event with RecordingEvent input."""
    # Update event type for ended event
    mock_recording_event_obj.event = "recording.ended"
    mock_recording_event_cls = type('RecordingEvent', (), {
        '__instancecheck__': lambda self, obj: isinstance(obj, MagicMock)
    })()
    
    with patch('app.models.recording.events.RecordingEvent', mock_recording_event_cls):
        from app.core.events.handlers.recording import handle_recording_ended
        await handle_recording_ended(mock_recording_event_obj)
        mock_recording_event_obj.save.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_recording_ended_error(mock_recording_event):
    """Test handling recording.ended event with error."""
    # Update event type for ended event
    mock_recording_event["event"] = "recording.ended"
    def raise_error(**kwargs):
        raise ValueError("Test error")
        
    mock_recording_event_cls = type('RecordingEvent', (), {
        '__call__': lambda self, **kwargs: raise_error(**kwargs),
        '__instancecheck__': lambda self, obj: False
    })()
    
    with patch('app.models.recording.events.RecordingEvent', mock_recording_event_cls):
        from app.core.events.handlers.recording import handle_recording_ended
        with pytest.raises(ValueError):
            await handle_recording_ended(mock_recording_event)
