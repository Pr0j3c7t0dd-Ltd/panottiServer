from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get configuration from environment
port = int(os.getenv("API_PORT", "8001"))
host = "0.0.0.0"
reload = True
