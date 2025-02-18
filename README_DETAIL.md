[![Discord Shield](https://discord.com/api/guilds/1316533461577891882/widget.png?style=shield)](https://discord.gg/fgVQWfPCmn)

# panottiServer

<img src="panotti-icon.png" alt="Panotti logo" width="64" height="64">

A FastAPI-based Python server for handling recording events with a plugin-based architecture and secure API endpoints.  More information, including the download link for the companionPanotti MacOS desktop app, can be found at [https://www.panotti.io/](https://www.panotti.io/).

Panotti was made by AI! Learn about AI in software development on our [Substack](https://pr0j3c7t0dd.substack.com/) — practical insights on leveraging AI in modern development workflows.

If you wish to support our work, please donate via PayPal:

[![Donate with PayPal](https://www.paypalobjects.com/en_US/GB/i/btn/btn_donateCC_LG.gif)](https://www.paypal.com/donate?hosted_button_id=NJTZJX5EN3F7N)

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
- Admin dashboard for monitoring and configuration
  - Real-time event monitoring
  - Plugin management interface
  - System configuration

## Help & Support

For comprehensive documentation, tutorials, and best practices, visit our official website at [https://www.panotti.io/](https://www.panotti.io/). Join our Discord community for support, discussions, and to connect with other users.

While this server is open source and compatible with any client that implements the API, it works best with the official Panotti MacOS desktop app, available on the MacOS App Store. The desktop app provides a seamless, integrated experience with features like:
- One-click setup and configuration
- Real-time recording status
- Callbacks to any server(s) (not just PanottiServer)
- Optional Google Calendar integration

## Installation

### Quick Setup (Recommended)

The easiest way to install panottiServer is using our automated install script:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Pr0j3c7t0dd-Ltd/panottiServer/refs/heads/main/install.sh)"
```

### Docker Setup (Preferred for Manual Installation)

For those who prefer to manually set up the server, the recommended method is to use our Docker setup script:

```bash
./scripts/docker_setup.py
```

This script will guide you through:
- Setting up Docker configuration
- Configuring environment variables
- Setting up SSL certificates
- Starting the Docker containers

### Local Setup (Advanced)

If you need to run the server directly on your machine without Docker, you can use the local setup script:

```bash
./scripts/local_setup.py
```

> ⚠️ **IMPORTANT**: Before running the setup script, please carefully review its contents at `scripts/local_setup.py`. This script will make changes to your system including installing dependencies and configuring your environment. Understanding these changes beforehand will help avoid any potential issues during installation.

The local setup script will automatically:
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

## Plugin Architecture

panottiServer features an extensible plugin architecture that allows you to create custom plugins to support your specific workflows. The server includes several built-in plugins:

- `audio_transcription_local`: Handles audio transcription using OpenAI Whisper
- `cleanup_files`: Manages file cleanup with safe deletion and notifications
- `desktop_notifier`: Provides desktop notifications for important events
- `events`: Core event system for plugin communication
- `meeting_notes_local`: Generates meeting notes using local Ollama LLM
- `meeting_notes_remote`: Generates meeting notes using remote LLM services
- `noise_reduction`: Audio preprocessing and noise reduction

## Requirements

- Python 3.12
- Rust (for FastAPI's Pydantic V2)
- Poetry (dependency management)
- Ollama (for meeting notes generation)
  - Default model: `llama3.1:8b` (other Ollama models are also supported)
  - Alternative: `deepseek-r1:8b` (reasoning LLM)
- OpenAI Whisper (for audio transcription)
- Minimum 8GB RAM available for Docker operations
- Docker with memory allocation:
  - Minimum: 12GB reserved, 24GB limit
  - Recommended: 16GB reserved, 32GB limit

System Dependencies (installed via setup script):
- `openai-whisper`: Audio transcription
- `terminal-notifier`: Desktop notifications (macOS)
- `ollama`: Local LLM processing

## Docker Deployment

The application consists of two main services:
1. Backend API server (FastAPI)
2. Admin frontend (React)

Both services are configured in the Docker Compose setup and will be built and started together.

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

The application uses these Docker Compose files:
- `docker-compose.yml`: Core services configuration (API + Admin Frontend)
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
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d
```

This will:
- Build and start the FastAPI backend (available at http://localhost:54789)
- Build and start the Admin frontend (available at http://localhost:54790)

2. Using Docker's Ollama (not recommended due to performance issues):
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

#### Accessing the Services

After starting the containers:
- Backend API: http://localhost:54789
- Admin Dashboard: http://localhost:54790
- API Documentation: http://localhost:54789/docs
- ReDoc Documentation: http://localhost:54789/redoc

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
./start_servers.sh
```

Both methods will read the port configuration from your `.env` file. The default port is 54789 if not specified.

Note: Using `uvicorn app.main:app --reload` directly will use port 54789 by default and won't read from the `.env` file.

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
- Swagger UI documentation at `http://localhost:54789/docs`
- ReDoc documentation at `http://localhost:54789/redoc`

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

To create a custom plugin:

1. Create a new directory in `app/plugins/`
2. Implement the plugin interface
3. Subscribe to relevant events using the EventBus
4. Add configuration in `plugin.yaml`

Example plugin structure:
```
your_plugin/
├── __init__.py
├── plugin.yaml    # Plugin configuration
├── models.py      # Data models
└── handlers.py    # Event handlers
```

### System Architecture

The application follows a modular, event-driven architecture:

```
app/
├── core/                     # Core system interfaces and protocols
│   ├── __init__.py
│   └── events/              # Event system implementation
│       ├── __init__.py      # Event system exports
│       ├── bus.py          # EventBus implementation
│       ├── models.py       # Event models
│       ├── persistence.py  # Event persistence
│       ├── types.py       # Type definitions
│       └── handlers/      # Event handlers
├── models/                  # Domain models
│   ├── __init__.py
│   ├── database.py         # Database functionality
│   └── recording/          # Recording-related models
│       ├── __init__.py
│       └── events.py       # Recording event models
├── plugins/                # Plugin implementations
│   ├── base.py            # Plugin base classes
│   ├── manager.py         # Plugin management
│   └── [plugin_name]/     # Individual plugins
└── utils/                 # Utility functions
    ├── logging_config.py  # Logging configuration
    └── directory_sync.py # Directory synchronization
```

### Event System

The event system is now part of the core package and uses a publish-subscribe pattern with the following components:

- **EventBus**: Robust asynchronous message broker for event distribution
- **Event**: Base class for all system events
- **EventContext**: Provides metadata and priority for each event
- **EventPriority**: Defines priority levels (LOW, NORMAL, HIGH)
- **EventHandler**: Base class for event handlers
- **EventPersistence**: Event storage and replay capabilities

Events are handled asynchronously with comprehensive error handling and logging.

### Creating a New Plugin

1. Create a new directory under `app/plugins/`
2. Create the plugin implementation:

```python
from app.core.plugins import PluginBase, PluginConfig
from app.core.events import EventBus, Event, EventHandler

class MyPlugin(PluginBase):
    def __init__(self, config: PluginConfig, event_bus: EventBus = None):
        super().__init__(config, event_bus)
        
    def handle_event(self, event: Event) -> None:
        # Event handling logic here
        pass
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
  - Uses llama3.1:8b model by default
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
./start_servers.sh
```

Both methods will read the port configuration from your `.env` file. The default port is 54789 if not specified.

Note: Using `uvicorn app.main:app --reload` directly will use port 54789 by default and won't read from the `.env` file.

#### Development
Run the server using uvicorn:
```bash
uvicorn app.main:app --reload
```

#### Production
For production deployment, use Gunicorn with Uvicorn workers:
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:54789
```

The server will start at `http://localhost:547890`

### Stopping the Server

To stop the server, you can use one of these methods:

1. If running in the foreground, press `Ctrl+C`
2. If running in the background or if `Ctrl+C` doesn't work, use:
```bash
pkill -f uvicorn
```

### API Documentation

Once the server is running, you can access:
- Swagger UI documentation at `http://localhost:547890/docs`
- ReDoc documentation at `http://localhost:54789/redoc`

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

- **Default**: `llama3.1:8b` - Good balance of quality and resource usage (minimum 24GB RAM)
- **Optional**: `llama3.3:70b` - Better quality notes but requires significant resources (minimum 80GB RAM)

Note: The model choice impacts the quality of:
- Meeting summaries
- Action item extraction
- Key point identification

Configure your preferred model in the meeting notes `plugin.yaml` file:
```yaml
model_name: "llama3.1:8b"  # or llama3.3:70b
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

## Warranty Disclaimer

THIS SOFTWARE IS PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE SOFTWARE IS WITH YOU. SHOULD THE SOFTWARE PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR, OR CORRECTION.

IN NO EVENT SHALL PR0J3CTTODD LTD BE LIABLE FOR ANY SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR INABILITY TO USE THE SOFTWARE.

For complete terms of use and privacy policy, please visit:
- Terms and Conditions: https://www.panotti.io/terms-and-conditions
- Privacy Policy: https://www.panotti.io/privacy-policy

Commercial licenses are available upon request.  Please visit [https://www.panotti.io/](https://www.panotti.io/) for more information.

 Copyright 2025 Pr0j3ctTodd Ltd. All rights reserved.