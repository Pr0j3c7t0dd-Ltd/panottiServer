# panottiServer

A FastAPI-based server for handling recording events with a plugin-based architecture and secure API endpoints.

## Features

- Plugin-based architecture for extensibility
- Event-driven system with structured logging
- Secure API endpoints with token authentication
- Dynamic plugin discovery and management
- Comprehensive test suite
- Swagger/OpenAPI documentation

## Architecture

The application follows a modular, plugin-based architecture:

```
app/
├── core/                     # Core system interfaces
├── models/                   # Domain models
├── plugins/                  # Plugin system
│   ├── base.py              # Base plugin classes
│   ├── manager.py           # Plugin lifecycle
│   └── [plugin_name]/       # Plugin directories
├── utils/                   # Utilities
└── tests/                   # Test suite
```

## Requirements

- Python 3.12
- Rust (for FastAPI's Pydantic V2)
- Poetry (dependency management)

## Installation

1. Install Rust (required for Pydantic V2's performance optimizations):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env  # Add Rust to your current shell session
```

2. Clone the repository:
```bash
git clone https://github.com/yourusername/panottiServer.git
cd panottiServer
```

3. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

4. Set up Python 3.12 using pyenv:
```bash
# Install pyenv
brew install pyenv

# Install Python 3.12 using pyenv
pyenv install 3.12

# Set local Python version for this project
pyenv local 3.12
```

5. Install dependencies using Poetry:
```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

6. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```
Then edit `.env` with your actual configuration values.

7. Set up HTTPS (optional):
```bash
# Create SSL directory
mkdir -p ssl

# Generate self-signed certificates
cd ssl
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/CN=localhost"
cd ..
```

Note: When using self-signed certificates in development, your browser will show a security warning. This is normal. For production, use certificates from a trusted certificate authority.

## Usage

### Starting the Server

There are two recommended ways to start the server:

1. Using the Python script:
```bash
python run_server.py
```

2. Using the shell script:
```bash
./start_server.sh
```

Both methods will read the port configuration from your `.env` file. The default port is 8001 if not specified.

Note: Using `uvicorn app.main:app --reload` directly will use port 8000 by default and won't read from the `.env` file.

#### Development
Run the server using uvicorn:
```bash
uvicorn app.main:app --reload
```

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. To run the pre-commit checks manually:

```bash
pre-commit run --all-files > output.txt 2>&1
```

This command will:
- Run all configured pre-commit hooks
- Save both standard output and error messages to `output.txt`
- Help identify any issues before committing code

Make sure to review `output.txt` for any warnings or errors that need to be addressed.

### Stopping the Server

To stop the server, you can use one of these methods:

1. If running in the foreground, press `Ctrl+C`
2. If running in the background or if `Ctrl+C` doesn't work, use:
```bash
pkill -f uvicorn
```

### API Documentation

Once the server is running, you can access:
- Swagger UI documentation at `http://localhost:8000/docs`
- ReDoc documentation at `http://localhost:8000/redoc`

### API Endpoints

#### Start Recording
```http
POST /start-recording
Header: X-API-Key: your_api_key
Content-Type: application/json

{
    "session_id": "unique_session_id",
    "timestamp": "2023-12-09T20:00:00Z"
}
```

#### End Recording
```http
POST /end-recording
Header: X-API-Key: your_api_key
Content-Type: application/json

{
    "session_id": "unique_session_id",
    "timestamp": "2023-12-09T20:10:00Z"
}
```

## Plugin Development

### Creating a New Plugin

1. Create a new directory under `app/plugins/`
2. Add `plugin.yaml` configuration:
```yaml
name: your_plugin_name
version: 1.0.0
description: Plugin description
enabled: true
dependencies: []
```

3. Implement `plugin.py`:
```python
from app.plugins.base import PluginBase

class YourPlugin(PluginBase):
    async def initialize(self):
        # Setup code
        pass

    async def start(self):
        # Start plugin
        pass

    async def stop(self):
        # Cleanup code
        pass
```

## Development

### Dependency Management

The project uses Poetry for dependency management. Here are some common commands:

```bash
# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Generate requirements.txt (useful for environments without Poetry)
poetry export -f requirements.txt --output requirements.txt

# Generate requirements.txt including development dependencies
poetry export -f requirements.txt --output requirements.txt --with dev
```

### Starting the Server

There are two recommended ways to start the server:

1. Using the Python script:
```bash
python run_server.py
```

2. Using the shell script:
```bash
./start_server.sh
```

Both methods will read the port configuration from your `.env` file. The default port is 8001 if not specified.

Note: Using `uvicorn app.main:app --reload` directly will use port 8000 by default and won't read from the `.env` file.

#### Development
Run the server using uvicorn:
```bash
uvicorn app.main:app --reload
```

#### Production
For production deployment, use Gunicorn with Uvicorn workers:
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

The server will start at `http://localhost:8000`

### Stopping the Server

To stop the server, you can use one of these methods:

1. If running in the foreground, press `Ctrl+C`
2. If running in the background or if `Ctrl+C` doesn't work, use:
```bash
pkill -f uvicorn
```

### API Documentation

Once the server is running, you can access:
- Swagger UI documentation at `http://localhost:8000/docs`
- ReDoc documentation at `http://localhost:8000/redoc`

### API Endpoints

#### Start Recording
```http
POST /start-recording
Header: X-API-Key: your_api_key
Content-Type: application/json

{
    "session_id": "unique_session_id",
    "timestamp": "2023-12-09T20:00:00Z"
}
```

#### End Recording
```http
POST /end-recording
Header: X-API-Key: your_api_key
Content-Type: application/json

{
    "session_id": "unique_session_id",
    "timestamp": "2023-12-09T20:10:00Z"
}
```

## Testing

Run the test suite:
```bash
pytest app/tests/
```

For test coverage:
```bash
pytest --cov=app app/tests/
```

## Logging

Logs are stored in both:
- Console output (for development)
- `logs/app.log` file (JSON formatted)

## Performance Notes

This implementation uses FastAPI with Pydantic V2, which requires Rust for its high-performance validation and serialization features. The Rust requirement enables:

- 5-50x faster validation compared to Pydantic V1
- Improved memory usage
- Better CPU utilization
- Enhanced type safety