import os
from datetime import UTC, datetime

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API configuration
API_PORT = os.getenv("API_PORT", "8001")  # Get API port from .env
BASE_URL = f"https://localhost:{API_PORT}"
API_KEY = os.getenv("API_KEY", "your_api_key_here")  # Get API key from .env
HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# SSL verification settings
SSL_VERIFY = False  # Set to True in production with valid certificates
if not SSL_VERIFY:
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def generate_recording_id() -> str:
    """Generate a recording ID with timestamp and random hex"""
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    random_hex = os.urandom(4).hex()[:8].upper()
    return f"{timestamp}_{random_hex}"


def test_basic_recording_flow() -> None:
    """Test recording flow with basic metadata"""
    recording_id = generate_recording_id()
    start_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Start recording
    start_response = requests.post(
        f"{BASE_URL}/api/recording-started",
        headers=HEADERS,
        json={
            "event": "Recording Started",
            "timestamp": start_time,
            "recordingId": recording_id,
        },
        verify=SSL_VERIFY,
        timeout=30,
    )
    print("\nStart Recording Response:", start_response.json())
    assert (
        start_response.status_code == 200
    ), f"Start recording failed: {start_response.text}"

    # End recording with basic metadata
    end_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    end_response = requests.post(
        f"{BASE_URL}/api/recording-ended",
        headers=HEADERS,
        json={
            "event": "Recording Ended",
            "timestamp": end_time,
            "recordingId": recording_id,
            "metadata": {"recordingDateTime": end_time},
        },
        verify=SSL_VERIFY,
        timeout=30,
    )
    print("\nEnd Recording Response:", end_response.json())
    assert end_response.status_code == 200, f"End recording failed: {end_response.text}"


def test_calendar_event_recording_flow() -> None:
    """Test recording flow with calendar event metadata"""
    recording_id = generate_recording_id()
    start_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Start recording
    start_response = requests.post(
        f"{BASE_URL}/api/recording-started",
        headers=HEADERS,
        json={
            "event": "Recording Started",
            "timestamp": start_time,
            "recordingId": recording_id,
        },
        verify=SSL_VERIFY,
        timeout=30,
    )
    print("\nStart Recording Response:", start_response.json())
    assert (
        start_response.status_code == 200
    ), f"Start recording failed: {start_response.text}"

    # End recording with calendar event metadata
    end_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    end_response = requests.post(
        f"{BASE_URL}/api/recording-ended",
        headers=HEADERS,
        json={
            "event": "Recording Ended",
            "timestamp": end_time,
            "recordingId": recording_id,
            "systemAudioPath": f"/path/to/system_{recording_id}.wav",
            "microphoneAudioPath": f"/path/to/mic_{recording_id}.wav",
            "metadata": {
                "eventTitle": "Pr0j3c7todd Companies house docs due",
                "eventProviderId": "2j11l4q826f3bfivj21gvfs3gp_20250501T070000Z",
                "eventProvider": "Google Calendar",
                "eventAttendees": ["todd@bluecliff.net", "todd@pr0j3c7t0dd.com"],
                "recordingDateTime": end_time,
            },
        },
        verify=SSL_VERIFY,
        timeout=30,
    )
    print("\nEnd Recording Response:", end_response.json())
    assert end_response.status_code == 200, f"End recording failed: {end_response.text}"


def test_invalid_recording_flow() -> None:
    """Test invalid recording scenarios"""
    # Test starting an already active recording
    recording_id = generate_recording_id()
    start_time = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # First start - should succeed
    start_response1 = requests.post(
        f"{BASE_URL}/api/recording-started",
        headers=HEADERS,
        json={
            "event": "Recording Started",
            "timestamp": start_time,
            "recordingId": recording_id,
        },
        verify=SSL_VERIFY,
        timeout=30,
    )
    assert start_response1.status_code == 200, "First start should succeed"

    # Second start with same ID - should fail
    start_response2 = requests.post(
        f"{BASE_URL}/api/recording-started",
        headers=HEADERS,
        json={
            "event": "Recording Started",
            "timestamp": start_time,
            "recordingId": recording_id,
        },
        verify=SSL_VERIFY,
        timeout=30,
    )
    assert start_response2.status_code == 400, "Second start should fail"

    # Test ending a non-existent recording
    non_existent_id = generate_recording_id()
    end_response = requests.post(
        f"{BASE_URL}/api/recording-ended",
        headers=HEADERS,
        json={
            "event": "Recording Ended",
            "timestamp": start_time,
            "recordingId": non_existent_id,
            "metadata": {"recordingDateTime": start_time},
        },
        verify=SSL_VERIFY,
        timeout=30,
    )
    assert end_response.status_code == 404, "Ending non-existent recording should fail"


if __name__ == "__main__":
    print("Testing basic recording flow...")
    test_basic_recording_flow()
    print("\nTesting calendar event recording flow...")
    test_calendar_event_recording_flow()
    print("\nTesting invalid recording flows...")
    test_invalid_recording_flow()
    print("\nAll tests completed!")
