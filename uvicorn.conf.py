import os

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get configuration from environment
port = int(os.getenv("API_PORT", "8001"))
host = os.getenv("API_HOST", "127.0.0.1")  # Default to localhost instead of 0.0.0.0
reload = True
