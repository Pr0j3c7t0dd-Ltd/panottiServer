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
- Minimum 8GB RAM available for Docker operations (model downloads and runtime)
- Docker with memory allocation:
  - Minimum: 12GB reserved, 24GB limit for running llama3.1:latest
  - Recommended: 16GB reserved, 32GB limit for better performance

Note: The memory requirements are primarily driven by the LLM model size and operations. The application uses Docker resource limits to manage memory allocation and prevent system instability.

## Installation

### Prerequisites

1. **Install Ollama** (for local meeting notes generation)
   - Download and install Ollama from the official website: [https://ollama.com/download](https://ollama.com/download)
   - This step must be completed before running the setup script
   - Do not use Homebrew for Ollama installation unless you specifically prefer it

2. **macOS Users**: You need Homebrew installed. If you don't have it, install it with:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

3. Ensure Python 3.12 is installed on your system:
```bash
python --version  # Should show Python 3.12.x
```

### Quick Setup (Recommended)

The easiest way to set up the application is to use the provided setup script. 

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

2. Install OpenAI Whisper (required for audio transcription):
```bash
brew install openai-whisper
```

3. Clone the repository:
```bash
git clone https://github.com/yourusername/panottiServer.git
cd panottiServer
```

4. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

5. Set up Python 3.12 using pyenv:
```bash
# Install pyenv
brew install pyenv

# Install Python 3.12 using pyenv
pyenv install 3.12

# Set local Python version for this project
pyenv local 3.12
```

6. Install dependencies using Poetry:
```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

7. Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```
Then edit `.env` with your actual configuration values.

8. Set up HTTPS (optional):
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

> ⚠️ **IMPORTANT WARNING**: Running Ollama within Docker is **strongly discouraged** due to significant resource constraints and potential stability issues. It is highly recommended to:
> 1. Run Ollama directly on your host machine
> 2. Run only the application in Docker
> 3. Configure the application to connect to your host's Ollama server
>
> This approach provides better performance and stability while maintaining the benefits of containerization for the main application.

#### Prerequisites

1. Install Docker:
```bash
# macOS (using Homebrew)
brew install docker docker-compose

# Linux
curl -fsSL https://get.docker.com | sh
```

2. Start Docker service (if not already running)

#### Configuration

The application uses two Docker Compose files:
- `docker-compose.yml`: Core application configuration
- `docker-compose.ollama.yml`: Optional Ollama service configuration

When using host's Ollama, update your plugin configuration in `app/plugins/meeting_notes_local/plugin.yaml`:
```yaml
ollama_url: "http://host.docker.internal:11434/api/generate"  # macOS/Windows
# or
ollama_url: "http://172.17.0.1:11434/api/generate"  # Linux
```

#### Running with Docker Compose

1. Using host's Ollama (recommended):
```bash
# Build and start the application
docker-compose up --build

# Run in detached mode
docker-compose up -d
```

2. Using Docker's Ollama (not recommended):
```bash
# Build and start with Ollama
docker-compose -f docker-compose.yml -f docker-compose.ollama.yml up --build

# Run in detached mode
docker-compose -f docker-compose.yml -f docker-compose.ollama.yml up -d
```

3. To stop the containers:
```bash
docker-compose down
```

#### Updating Docker Containers

When you need to update the containers with new changes:

1. Stop the running containers:
```bash
docker compose down
```

2. Rebuild the containers without using cache:
```bash
# For host Ollama (recommended):
docker compose build --no-cache app

# For Docker Ollama (not recommended):
docker compose -f docker-compose.yml -f docker-compose.ollama.yml build --no-cache
```

3. Start the containers again:
```bash
# For host Ollama (recommended):
docker compose up -d

# For Docker Ollama (not recommended):
docker compose -f docker-compose.yml -f docker-compose.ollama.yml up -d
```

All in one command:
```bash
# For host Ollama (recommended):
docker compose down && docker compose build --no-cache app && docker compose up -d

# For Docker Ollama (not recommended):
docker compose down && docker compose -f docker-compose.yml -f docker-compose.ollama.yml build --no-cache && docker compose -f docker-compose.yml -f docker-compose.ollama.yml up -d
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

### Directory Synchronization

The application includes a directory synchronization feature that can monitor and automatically copy files from source directories to destination directories. The primary use case is to automatically copy generated meeting notes from the application's output directories (`data/meeting_notes_remote` or `data/meeting_notes_local`) to other locations on your system for backup or further processing.

#### Configuration

1. In your `.env` file, enable directory sync:
```bash
DIRECTORY_SYNC_ENABLED=true
```

2. Configure directory pairs to monitor using the `DIRECTORY_SYNC_PAIRS` environment variable. For example, to sync meeting notes:
```bash
DIRECTORY_SYNC_PAIRS='[{"source": "data/meeting_notes_remote", "destination": "/path/to/your/notes/backup"}, {"source": "data/meeting_notes_local", "destination": "/path/to/your/local/notes"}]'
```

Notes:
- Paths can be absolute or relative to the application root
- Multiple directory pairs can be monitored simultaneously
- The feature can be disabled by setting `DIRECTORY_SYNC_ENABLED=false`
- Source directories are monitored recursively
- Files are copied with their metadata preserved

#### Docker Considerations

When running the application in Docker, to sync directories between your host machine and the container:

1. First add a volume mount in your `docker-compose.yml`:
```yaml
volumes:
  - /path/on/host:/path/in/container
```

2. Then configure the sync in your `.env` file using the container paths:
```env
DIRECTORY_SYNC_ENABLED=true
DIRECTORY_SYNC_PAIRS='[{"source": "/path/in/container", "destination": "/other/path/in/container"}]'
```

Note: Always use the paths as they appear inside the container when configuring `DIRECTORY_SYNC_PAIRS`, not the host paths.

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

- **Default**: `llama3.1:latest` - Good balance of quality and resource usage (minimum 24GB RAM)
- **Optional**: `llama3.3:70b` - Better quality notes but requires significant resources (minimum 80GB RAM)

Note: The model choice impacts the quality of:
- Meeting summaries
- Action item extraction
- Key point identification

Configure your preferred model in the meeting notes `plugin.yaml` file:
```yaml
model_name: "llama3.1:latest"  # or llama3.3:70b
```

Important memory considerations:
- Initial model download requires at least 8GB of available system memory
- Runtime memory usage varies based on the model and context length
- Docker memory limits should be set according to your chosen model
- System should have enough free memory to handle both model operations and other processes

## Package Version Management

### Upgrading All Packages

To upgrade all Python packages to their latest versions:

```bash
pip install --upgrade $(pip freeze | sed 's/==.*//g')
```

After upgrading, update your requirements.txt:

```bash
pip freeze > requirements.txt
```

Note: Be sure to test your application thoroughly after upgrading packages as new versions may introduce breaking changes.