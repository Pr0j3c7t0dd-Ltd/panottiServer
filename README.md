# panottiServer

A FastAPI-based server for handling recording events with a plugin-based architecture and secure API endpoints.

## Features

- Plugin-based architecture for extensibility
- Event-driven system with structured logging
- Secure API endpoints with token authentication
- Dynamic plugin discovery and management
- Comprehensive test suite
- Swagger/OpenAPI documentation
- Audio transcription with OpenAI Whisper (offline mode)
- Automated meeting notes generation with local or remote Ollama LLM
- Desktop notifications for important events

## Architecture

The application follows a modular, plugin-based architecture:

```
app/
├── core/                     # Core system interfaces
├── models/                   # Domain models
├── plugins/                  # Plugin system
│   ├── base.py              # Base plugin classes
│   ├── manager.py           # Plugin lifecycle
│   ├── audio_transcription/ # Audio transcription plugin
│   ├── meeting_notes_local/ # Local meeting notes plugin
│   ├── meeting_notes_remote/# Remote meeting notes plugin
│   └── desktop_notifier/    # Desktop notifications plugin
├── utils/                   # Utilities
└── tests/                   # Test suite
```

## Requirements

- Python 3.12
- Rust (for FastAPI's Pydantic V2)
- Poetry (dependency management)
- Ollama (for meeting notes generation)
- OpenAI Whisper (for audio transcription)

## Installation

### Quick Setup (Recommended)

The easiest way to set up the application is to use the provided setup script. 

#### Prerequisites

1. **macOS Users**: You need Homebrew installed. If you don't have it, install it with:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Ensure Python 3.12 is installed on your system:
```bash
python --version  # Should show Python 3.12.x
```

#### Running the Setup

1. Run the setup script:
```bash
./scripts/setup.py
```

The setup script will automatically:
- Verify/install Rust (required for Pydantic V2)
- Verify/install Poetry for dependency management
- Install system dependencies via Homebrew (macOS):
  - openai-whisper (for audio transcription)
  - terminal-notifier (for desktop notifications)
  - ollama (for local LLM processing)
- Set up the Python virtual environment
- Install all dependencies
- Configure environment files
- Set up plugin configurations
- Download required ML models
- Generate SSL certificates

### Manual Setup (Advanced)

If you prefer to set up components manually, follow these steps:

1. Install Rust (required for Pydantic V2's performance optimizations):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env  # Add Rust to your current shell session
```

2. Install Ollama (required for meeting notes generation):
```bash
curl https://ollama.ai/install.sh | sh
```

3. Install OpenAI Whisper (required for audio transcription):
```bash
brew install openai-whisper
```

4. Clone the repository:
```bash
git clone https://github.com/yourusername/panottiServer.git
cd panottiServer
```

5. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

6. Set up Python 3.12 using pyenv:
```bash
# Install pyenv
brew install pyenv

# Install Python 3.12 using pyenv
pyenv install 3.12

# Set local Python version for this project
pyenv local 3.12
```

7. Install dependencies using Poetry:
```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

8. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```
Then edit `.env` with your actual configuration values.

9. Set up HTTPS (optional):
```bash
# Create SSL directory
mkdir -p ssl

# Generate self-signed certificates
cd ssl
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/CN=localhost"
cd ..
```

Note: When using self-signed certificates in development, your browser will show a security warning. This is normal. For production, use certificates from a trusted certificate authority.

### Docker Deployment

The application can be run using Docker and Docker Compose for easier deployment and consistent environments.

#### Prerequisites

1. Install Docker:
```bash
# macOS (using Homebrew)
brew install docker docker-compose

# Linux
curl -fsSL https://get.docker.com | sh
```

2. Start Docker service (if not already running)

#### Running with Docker Compose

1. Build and start the containers:
```bash
docker-compose up --build
```

2. To run in detached mode (background):
```bash
docker-compose up -d
```

3. To stop the containers:
```bash
docker-compose down
```

#### Running with Docker directly

1. Build the Docker image:
```bash
docker build -t panotti-server .
```

2. Run the container:
```bash
docker run -p 8001:8001 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  panotti-server
```

The server will be accessible at `http://localhost:8001`

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

### Available Plugins

#### Audio Transcription
- Transcribes WAV audio files with timestamps
- Uses OpenAI's Whisper model through `faster-whisper` library
- Supports concurrent processing
- Merges multiple transcripts based on timestamps

#### Meeting Notes (Local)
- Generates meeting notes from transcripts using local Ollama LLM
- Listens for transcription completion events
- Produces structured markdown notes with:
  - Meeting title and information
  - Executive summary
  - Key discussion points
  - Action items
  - Decisions made
  - Next steps

#### Meeting Notes (Remote)
- Same features as local meeting notes plugin
- Connects to remote Ollama instance
- Configurable model and parameters
- Supports larger context windows

#### Desktop Notifier
- Provides desktop notifications for important events
- Customizable notification settings
- Cross-platform support

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

## Model Requirements

### Meeting Notes Generation

For optimal meeting notes generation, this application requires a large language model. We recommend:

- **Recommended**: `llama3.3:70b` - Best quality notes, requires significant GPU resources (minimum 80GB VRAM)
- **Alternative**: `llama3.1:latest` - Acceptable quality, lower resource requirements (minimum 24GB VRAM)

Note: Using smaller models may result in reduced quality of meeting notes and summaries. The model choice significantly impacts the quality of:
- Meeting summaries
- Action item extraction
- Key point identification

Configure your preferred model in the meeting notes `plugin.yaml` file:
```bash
model_name: "llama3.3:70b"  # or llama3.1:latest
```