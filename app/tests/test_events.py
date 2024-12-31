import os
from dataclasses import asdict
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from ..main import RecordingEndRequest, RecordingStartRequest, app

client = TestClient(app)

# Mock API key for testing
TEST_API_KEY = "test_api_key"
os.environ["API_KEY"] = TEST_API_KEY


def test_start_recording_success():
    request = RecordingStartRequest(
        session_id="test_session_1", timestamp=datetime.now(UTC)
    )
    response = client.post(
        "/start-recording", headers={"X-API-Key": TEST_API_KEY}, json=asdict(request)
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_start_recording_duplicate_session():
    session_id = "test_session_2"
    timestamp = datetime.now(UTC)

    # Start first recording
    request1 = RecordingStartRequest(session_id=session_id, timestamp=timestamp)
    response1 = client.post(
        "/start-recording", headers={"X-API-Key": TEST_API_KEY}, json=asdict(request1)
    )
    assert response1.status_code == 200

    # Try to start second recording with same session_id
    request2 = RecordingStartRequest(session_id=session_id, timestamp=timestamp)
    response2 = client.post(
        "/start-recording", headers={"X-API-Key": TEST_API_KEY}, json=asdict(request2)
    )
    assert response2.status_code == 400


def test_end_recording_success():
    session_id = "test_session_3"
    start_time = datetime.now(UTC)

    # Start recording
    request1 = RecordingStartRequest(session_id=session_id, timestamp=start_time)
    response1 = client.post(
        "/start-recording", headers={"X-API-Key": TEST_API_KEY}, json=asdict(request1)
    )
    assert response1.status_code == 200

    # End recording
    end_time = datetime.now(UTC)
    request2 = RecordingEndRequest(session_id=session_id, timestamp=end_time)
    response2 = client.post(
        "/end-recording", headers={"X-API-Key": TEST_API_KEY}, json=asdict(request2)
    )
    assert response2.status_code == 200
    assert "duration_seconds" in response2.json()


def test_end_recording_invalid_session():
    request = RecordingEndRequest(
        session_id="nonexistent_session", timestamp=datetime.now(UTC)
    )
    response = client.post(
        "/end-recording", headers={"X-API-Key": TEST_API_KEY}, json=asdict(request)
    )
    assert response.status_code == 400


def test_invalid_api_key():
    request = RecordingStartRequest(
        session_id="test_session", timestamp=datetime.now(UTC)
    )
    response = client.post(
        "/start-recording", headers={"X-API-Key": "invalid_key"}, json=asdict(request)
    )
    assert response.status_code == 403
