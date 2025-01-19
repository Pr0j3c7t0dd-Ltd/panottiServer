from datetime import datetime, UTC
import pytest
from pydantic import ValidationError

from app.models.recording.requests import RecordingEndRequest


def test_recording_end_request_minimal():
    """Test creating RecordingEndRequest with only required fields"""
    request = RecordingEndRequest(recording_id="test123")
    assert request.recording_id == "test123"
    assert isinstance(request.end_time, datetime)
    assert request.metadata is None
    assert request.status == "completed"
    assert request.error_message is None


def test_recording_end_request_full():
    """Test creating RecordingEndRequest with all fields"""
    end_time = datetime.now(UTC)
    metadata = {"key": "value"}
    request = RecordingEndRequest(
        recording_id="test123",
        end_time=end_time,
        metadata=metadata,
        status="failed",
        error_message="Something went wrong"
    )
    assert request.recording_id == "test123"
    assert request.end_time == end_time
    assert request.metadata == metadata
    assert request.status == "failed"
    assert request.error_message == "Something went wrong"


def test_recording_end_request_missing_required():
    """Test that RecordingEndRequest requires recording_id"""
    with pytest.raises(ValidationError) as exc_info:
        RecordingEndRequest()
    
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("recording_id",)
    assert errors[0]["type"] == "missing" 