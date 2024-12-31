import os
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.main import app
from app.models.event import RecordingEndRequest, RecordingStartRequest

client = TestClient(app)

# Mock API key for testing
TEST_API_KEY = "test_api_key"
os.environ["API_KEY"] = TEST_API_KEY


def test_start_recording_success() -> None:
    timestamp = datetime.now(UTC).isoformat()
    request = RecordingStartRequest(
        recording_id="test_session_1", timestamp=timestamp, metadata={"test": "data"}
    )
    response = client.post(
        "/start-recording",
        headers={"X-API-Key": TEST_API_KEY},
        json=request.model_dump(),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_start_recording_duplicate_session() -> None:
    session_id = "test_session_2"
    timestamp = datetime.now(UTC).isoformat()

    # Start first recording
    request1 = RecordingStartRequest(
        recording_id=session_id, timestamp=timestamp, metadata={"test": "data"}
    )
    response1 = client.post(
        "/start-recording",
        headers={"X-API-Key": TEST_API_KEY},
        json=request1.model_dump(),
    )
    assert response1.status_code == 200

    # Try to start second recording with same session_id
    request2 = RecordingStartRequest(
        recording_id=session_id, timestamp=timestamp, metadata={"test": "data"}
    )
    response2 = client.post(
        "/start-recording",
        headers={"X-API-Key": TEST_API_KEY},
        json=request2.model_dump(),
    )
    assert response2.status_code == 400


def test_end_recording_success() -> None:
    session_id = "test_session_3"
    start_time = datetime.now(UTC).isoformat()

    # Start recording
    request1 = RecordingStartRequest(
        recording_id=session_id, timestamp=start_time, metadata={"test": "data"}
    )
    response1 = client.post(
        "/start-recording",
        headers={"X-API-Key": TEST_API_KEY},
        json=request1.model_dump(),
    )
    assert response1.status_code == 200

    # End recording
    end_time = datetime.now(UTC).isoformat()
    request2 = RecordingEndRequest(
        recording_id=session_id,
        timestamp=end_time,
        system_audio_path="/path/to/system.wav",
        microphone_audio_path="/path/to/mic.wav",
        metadata={"test": "data"},
    )
    response2 = client.post(
        "/end-recording",
        headers={"X-API-Key": TEST_API_KEY},
        json=request2.model_dump(),
    )
    assert response2.status_code == 200
    assert "duration_seconds" in response2.json()


def test_end_recording_invalid_session() -> None:
    timestamp = datetime.now(UTC).isoformat()
    request = RecordingEndRequest(
        recording_id="nonexistent_session",
        timestamp=timestamp,
        system_audio_path="/path/to/system.wav",
        microphone_audio_path="/path/to/mic.wav",
        metadata={"test": "data"},
    )
    response = client.post(
        "/end-recording", headers={"X-API-Key": TEST_API_KEY}, json=request.model_dump()
    )
    assert response.status_code == 400


def test_invalid_api_key() -> None:
    timestamp = datetime.now(UTC).isoformat()
    request = RecordingStartRequest(
        recording_id="test_session", timestamp=timestamp, metadata={"test": "data"}
    )
    response = client.post(
        "/start-recording",
        headers={"X-API-Key": "invalid_key"},
        json=request.model_dump(),
    )
    assert response.status_code == 403
