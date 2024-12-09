import uvicorn
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get port from environment
port = int(os.getenv("API_PORT", "8001"))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
