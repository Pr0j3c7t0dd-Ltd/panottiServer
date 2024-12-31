import os
import pathlib

import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Get port from environment
    port = int(os.getenv("API_PORT", "8001"))
    
    # Get host from environment
    host = os.getenv("API_HOST", "127.0.0.1")  # Default to localhost

    # Get SSL certificate paths
    base_dir = pathlib.Path(__file__).parent
    ssl_keyfile = str(base_dir / "ssl" / "key.pem")
    ssl_certfile = str(base_dir / "ssl" / "cert.pem")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
    )

if __name__ == "__main__":
    main()
