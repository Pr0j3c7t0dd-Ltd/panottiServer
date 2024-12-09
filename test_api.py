import requests
import json
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# API configuration
API_PORT = os.getenv("API_PORT", "8001")  # Get API port from .env
BASE_URL = f"http://localhost:{API_PORT}"
API_KEY = os.getenv("API_KEY", "your_api_key_here")  # Get API key from .env
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def test_recording_flow():
    # Start recording
    session_id = "test_session"
    start_time = datetime.now(timezone.utc)
    
    start_response = requests.post(
        f"{BASE_URL}/start-recording",
        headers=HEADERS,
        json={
            "session_id": session_id,
            "timestamp": start_time.isoformat(),
            "type": "start"
        }
    )
    print("\nStart Recording Response:", start_response.json())
    
    # End recording
    end_time = datetime.now(timezone.utc)
    end_response = requests.post(
        f"{BASE_URL}/end-recording",
        headers=HEADERS,
        json={
            "session_id": session_id,
            "timestamp": end_time.isoformat(),
            "type": "end"
        }
    )
    print("\nEnd Recording Response:", end_response.json())

if __name__ == "__main__":
    test_recording_flow()
