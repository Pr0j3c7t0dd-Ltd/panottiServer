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

## Help & Support

For comprehensive documentation, tutorials, and best practices, visit our official website at [https://www.panotti.io/](https://www.panotti.io/). Join our Discord community for support, discussions, and to connect with other users.

While this server is open source and compatible with any client that implements the API, it works best with the official Panotti MacOS desktop app, available on the MacOS App Store. The desktop app provides a seamless, integrated experience with features like:
- One-click setup and configuration
- Real-time recording status
- Callbacks to any server(s) (not just PanottiServer)
- Optional Google Calendar integration

## Plugin Architecture

panottiServer features an extensible plugin architecture that allows you to create custom plugins to support your specific workflows. Whether you need to integrate with your team's tools, add custom processing steps, or implement unique features, you can easily extend the server's functionality through plugins.

Check out the [Plugin Development](#plugin-development) section below for a guide on creating your own plugins. In the future, we'll launch a community plugin repository where users can share and discover plugins built by the Panotti community.

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

### System Architecture

The application follows a modular, plugin-based architecture:

```
app/
├── core/                     # Core system interfaces and protocols
│   ├── __init__.py
│   ├── events/              # Event system interfaces
│   │   ├── __init__.py     # EventBus protocol
│   │   └── models.py       # Core event models
│   └── plugins/            # Plugin system core
│       ├── __init__.py     # Plugin exports
│       ├── interface.py    # Plugin base class and config
│       ├── manager.py      # Plugin lifecycle management
│       └── protocol.py     # Plugin protocols and interfaces
├── plugins/                # Plugin implementations
│   ├── __init__.py
│   ├── example/           # Example plugin
│   │   ├── __init__.py
│   │   ├── plugin.py
│   │   └── plugin.yaml
│   └── ...                # Other plugins
```

### Creating a New Plugin

1. Create a new directory under `app/plugins/`
2. Create the plugin implementation:

```python
from app.core.plugins import PluginBase, PluginConfig

class MyPlugin(PluginBase):
    def __init__(self, config: PluginConfig, event_bus=None):
        super().__init__(config, event_bus)
```

### Available Plugins

#### Noise Reduction
- Reduces background noise in audio recordings
- Advanced signal processing with FFT-based alignment
- Configurable parameters for noise reduction
- Supports both time and frequency domain processing
- Optimized for speech clarity

#### Audio Transcription Local
- Transcribes WAV audio files with timestamps using Whisper locally
- Uses OpenAI's Whisper model through `faster-whisper` library
- Supports concurrent processing
- Configurable model selection from tiny to large
- Automatic transcript cleanup, formatting and merging

#### Meeting Notes Local
- Generates meeting notes from transcripts using local Ollama LLM
- Listens for transcription completion events
- Produces structured markdown notes with:
  - Meeting title and information
  - Executive summary
  - Key discussion points
  - Action items
  - Decisions made
  - Next steps

#### Meeting Notes Remote
- Same features as local meeting notes plugin
- Connects to remote OpenAI, Anthropic, or Google APIs to generate notes
- Configurable model and parameters
- Supports larger context windows

#### Desktop Notifier
- Provides desktop notifications for important events
- Customizable notification settings
- Cross-platform support
- Auto-opens generated notes
- Concurrent notification handling

#### Cleanup Files
- Automated file management and cleanup
- Configurable include/exclude directories
- Safe deletion with notifications
- Protects important directories
- Integration with desktop notifications

#### Example Plugin
- Reference implementation for plugin development
- Demonstrates best practices and patterns
- Shows event handling and configuration
- Includes comprehensive documentation

### Default Plugin Configuration

The server comes with several built-in plugins. Here's a summary of each plugin and its default state:

#### Audio Transcription Local
- **Status**: Enabled by default
- **Dependencies**: noise_reduction
- **Description**: Transcribes audio using Whisper's base.en model locally
- **Features**:
  - Configurable output directory for transcripts
  - Concurrent task processing
  - Transcript cleanup option

#### Meeting Notes Local
- **Status**: Enabled by default
- **Dependencies**: audio_transcription
- **Description**: Generates meeting notes using local Ollama LLM
- **Features**:
  - Uses llama3.1:latest model by default
  - Configurable Ollama URL for Docker/local setup
  - Large context window (131K tokens)

#### Meeting Notes Remote
- **Status**: Disabled by default
- **Dependencies**: audio_transcription
- **Description**: Generates meeting notes using remote LLM services
- **Features**:
  - Supports OpenAI, Anthropic, and Google providers
  - Configurable API keys and models
  - Timeout settings for long meetings

#### Noise Reduction
- **Status**: Enabled by default
- **Dependencies**: None
- **Description**: Reduces background noise in audio recordings
- **Features**:
  - Configurable noise reduction parameters
  - FFT-based alignment
  - Frequency domain processing

#### Desktop Notifier
- **Status**: Enabled by default
- **Dependencies**: meeting_notes_local, meeting_notes_remote
- **Description**: Provides system notifications for important events
- **Features**:
  - Auto-opens generated notes
  - Concurrent notification handling

#### Cleanup Files
- **Status**: Disabled by default
- **Dependencies**: desktop_notifier
- **Description**: Manages automatic cleanup of processed files
- **Features**:
  - Configurable include/exclude directories
  - Safe deletion with notifications
  - Protects important directories by default

#### Example Plugin
- **Status**: Disabled by default
- **Dependencies**: None
- **Description**: Reference implementation for plugin development
- **Features**:
  - Demonstrates plugin structure
  - Shows event handling patterns
  - Includes development best practices

### Default Pluging Workflow

1. End recording
2. Audio cleaning
3. Audio transcription
4. Meeting notes (local or remote or both)
5. Desktop notification(s)
6. Cleanup files

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

© 2025 Pr0j3ctTodd Ltd. All rights reserved.