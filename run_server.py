import uvicorn
from dotenv import load_dotenv
import os
import pathlib

# Load environment variables
load_dotenv()

# Get port from environment
port = int(os.getenv("API_PORT", "8001"))

# Get SSL certificate paths
base_dir = pathlib.Path(__file__).parent
ssl_keyfile = str(base_dir / "ssl" / "key.pem")
ssl_certfile = str(base_dir / "ssl" / "cert.pem")

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile
    )
