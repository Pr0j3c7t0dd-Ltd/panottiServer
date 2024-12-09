import requests
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
import os
import uuid

# Load environment variables
load_dotenv()

# API configuration
API_PORT = os.getenv("API_PORT", "8001")  # Get API port from .env
BASE_URL = f"https://localhost:{API_PORT}"
API_KEY = os.getenv("API_KEY", "your_api_key_here")  # Get API key from .env
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

# SSL verification settings
SSL_VERIFY = False  # Set to True in production with valid certificates
if not SSL_VERIFY:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def test_recording_flow():
    # Generate a unique recording ID
    recording_id = str(uuid.uuid4())
    
    # Start recording
    start_time = "2024-12-09T22:17:07Z"  # Using example timestamp
    
    start_response = requests.post(
        f"{BASE_URL}/api/recording-started",
        headers=HEADERS,
        json={
            "event": "Recording Started",
            "timestamp": start_time,
            "recordingId": recording_id
        },
        verify=SSL_VERIFY
    )
    print("\nStart Recording Response:", start_response.json())
    assert start_response.status_code == 200, f"Start recording failed: {start_response.text}"
    
    # End recording
    end_time = "2024-12-09T22:18:07Z"  # Using example timestamp
    system_audio_path = f"/recordings/{recording_id}_system_audio.wav"
    microphone_audio_path = f"/recordings/{recording_id}_microphone.wav"
    
    end_response = requests.post(
        f"{BASE_URL}/api/recording-ended",
        headers=HEADERS,
        json={
            "event": "Recording Ended",
            "timestamp": end_time,
            "recordingId": recording_id,
            "systemAudioPath": system_audio_path,
            "MicrophoneAudioPath": microphone_audio_path
        },
        verify=SSL_VERIFY
    )
    print("\nEnd Recording Response:", end_response.json())
    assert end_response.status_code == 200, f"End recording failed: {end_response.text}"

def test_invalid_recording_flow():
    # Test starting an already active recording
    recording_id = str(uuid.uuid4())
    start_time = "2024-12-09T22:17:07Z"
    
    # First start - should succeed
    start_response1 = requests.post(
        f"{BASE_URL}/api/recording-started",
        headers=HEADERS,
        json={
            "event": "Recording Started",
            "timestamp": start_time,
            "recordingId": recording_id
        },
        verify=SSL_VERIFY
    )
    assert start_response1.status_code == 200, "First start should succeed"
    
    # Second start with same ID - should fail
    start_response2 = requests.post(
        f"{BASE_URL}/api/recording-started",
        headers=HEADERS,
        json={
            "event": "Recording Started",
            "timestamp": start_time,
            "recordingId": recording_id
        },
        verify=SSL_VERIFY
    )
    print("\nDuplicate Start Response:", start_response2.json())
    assert start_response2.status_code == 400, "Second start should fail"
    
    # Test ending a non-existent recording
    non_existent_id = str(uuid.uuid4())
    end_response = requests.post(
        f"{BASE_URL}/api/recording-ended",
        headers=HEADERS,
        json={
            "event": "Recording Ended",
            "timestamp": "2024-12-09T22:18:07Z",
            "recordingId": non_existent_id,
            "systemAudioPath": f"/recordings/{non_existent_id}_system_audio.wav",
            "MicrophoneAudioPath": f"/recordings/{non_existent_id}_microphone.wav"
        },
        verify=SSL_VERIFY
    )
    print("\nNon-existent Recording End Response:", end_response.json())
    assert end_response.status_code == 400, "Ending non-existent recording should fail"

if __name__ == "__main__":
    print("Testing normal recording flow...")
    test_recording_flow()
    print("\nTesting invalid recording flows...")
    test_invalid_recording_flow()
    print("\nAll tests completed!")
