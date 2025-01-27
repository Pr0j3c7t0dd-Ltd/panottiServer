# PanottiServer Architecture Documentation

## Overview

PanottiServer is a modular, event-driven system designed for audio processing and meeting transcription. The architecture emphasizes extensibility, maintainability, and robust error handling through a plugin-based system.

## Directory Structure

```
app/
├── core/                     # Core system interfaces and protocols
│   ├── events/              # Event system implementation
│   │   ├── bus.py          # EventBus implementation
│   │   ├── models.py       # Event models
│   │   ├── persistence.py  # Event persistence
│   │   ├── types.py        # Type definitions
│   │   └── handlers/       # Event handlers
│   └── plugins/            # Plugin system core
│       ├── interface.py    # Plugin interfaces
│       ├── manager.py      # Plugin management
│       └── protocol.py     # Plugin protocols
├── models/                  # Domain models
│   ├── database.py         # Database functionality
│   ├── migrations/         # Database migrations
│   └── recording/          # Recording-related models
│       ├── events.py       # Recording event models
│       └── requests.py     # API request models
├── plugins/                # Plugin implementations
│   ├── audio_transcription_local/  # Local transcription
│   ├── cleanup_files/             # File cleanup
│   └── desktop_notifier/          # Desktop notifications
└── main.py                # Application entry point
```

## Core Components

### Event System

The event system is the backbone of PanottiServer, facilitating communication between components:

- **EventBus** (`core/events/bus.py`): Asynchronous event distribution
- **Event Models** (`core/events/models.py`): Type-safe event definitions
- **Event Persistence** (`core/events/persistence.py`): Event storage and replay
- **Event Types** (`core/events/types.py`): Type definitions and constants
- **Event Handlers** (`core/events/handlers/`): Event processing logic

Event Format Example:
```python
Event.create(
    name="transcription_local.completed",
    data={
        "recording": {...},
        "transcription": {
            "status": status,
            "timestamp": timestamp,
            "recording_id": recording_id,
            "output_file": output_file,
            "transcript_paths": transcript_paths,
            "model": model,
            "language": language,
            "speaker_labels": {...}
        },
        "metadata": metadata,
        "context": {
            "correlation_id": correlation_id,
            "source_plugin": source_plugin,
            "metadata": metadata
        }
    },
    correlation_id=correlation_id,
    source_plugin=source_plugin,
    priority=EventPriority.NORMAL
)
```

### Plugin System

The plugin system enables extensible functionality:

- **Plugin Interface** (`core/plugins/interface.py`): Base plugin definitions
- **Plugin Manager** (`core/plugins/manager.py`): Dynamic loading and lifecycle
- **Plugin Protocol** (`core/plugins/protocol.py`): Plugin contracts

Key Features:
- Dynamic discovery and loading
- Configuration via YAML files
- Lifecycle management
- Error isolation

### Models

Domain models represent business entities:

- **Database** (`models/database.py`): Database operations
- **Recording Models** (`models/recording/`): Audio processing models
- **Migrations** (`models/migrations/`): Database schema evolution

## Technology Stack

- **Framework**: FastAPI
- **Database**: SQLite/PostgreSQL
- **Event System**: Custom EventBus
- **Configuration**: YAML + Environment Variables
- **Development Tools**:
  - Poetry (dependency management)
  - Black (formatting)
  - Ruff (linting)
  - MyPy (type checking)
  - pytest (testing)

## Core Plugins

### Audio Transcription Local
- Local speech-to-text processing
- Multiple model support
- Noise reduction
- Speaker detection

### Cleanup Files
- Automatic file management
- Resource cleanup
- Storage optimization

### Desktop Notifier
- System notifications
- Process status updates
- Error alerts

## Security

1. **Authentication**
   - API key validation
   - Role-based access
   - Secure credential storage

2. **Data Protection**
   - Environment-based configuration
   - Data encryption
   - Access control
   - Audit logging

3. **Error Handling**
   - Safe error messages
   - Secure logging
   - Resource cleanup
   - Failure isolation

## Development Guidelines

1. **Code Style**
   - Follow PEP 8
   - Use type hints
   - Document with docstrings
   - Maintain consistent formatting

2. **Testing**
   - Write unit tests
   - Include integration tests
   - Mock external services
   - Test plugins independently

3. **Error Handling**
   - Use structured logging
   - Implement graceful degradation
   - Provide error context
   - Clean error recovery

4. **Plugin Development**
   - Follow plugin interface
   - Include configuration example
   - Document dependencies
   - Handle lifecycle events

## Event System Implementation

### Event Bus
```python
class EventBus:
    """Central event dispatcher with async support."""
    def __init__(self):
        self._queue = asyncio.Queue(maxsize=1000)  # Backpressure control
        
    async def publish(self, event: Event) -> None:
        try:
            await asyncio.wait_for(self._queue.put(event), timeout=5.0)
        except asyncio.TimeoutError:
            # Handle backpressure
            pass

    async def subscribe(self, handler: EventHandler) -> None:
        """Register event handler."""
        
    async def unsubscribe(self, handler: EventHandler) -> None:
        """Remove event handler."""
```

### Event Correlation
```python
def create_child_event(self, parent_event: Event, name: str, data: dict) -> Event:
    """Create a new event that maintains correlation with parent."""
    return Event(
        name=name,
        data=data,
        correlation_id=parent_event.correlation_id,
        source_plugin=self.name
    )
```

## Plugin Development

### Plugin Lifecycle
1. **Registration**
   - Plugin discovery
   - Dependency validation
   - Configuration loading
   - Event subscription

2. **Initialization**
   - Resource allocation
   - State initialization
   - Connection establishment
   - Health check setup

3. **Operation**
   - Event processing
   - State management
   - Error handling
   - Health monitoring

4. **Cleanup**
   - Resource release
   - Connection cleanup
   - State persistence
   - Graceful shutdown

### State Management
```python
class StatefulPlugin(BasePlugin):
    def __init__(self):
        self._state = {}
        self._lock = asyncio.Lock()
        
    async def update_state(self, key: str, value: Any) -> None:
        async with self._lock:
            self._state[key] = value
```

## Deployment

1. **Environment Setup**
   - Configure environment variables
   - Install dependencies
   - Setup plugins
   - Initialize database

2. **Monitoring**
   - Structured logging
   - Error tracking
   - Performance monitoring
   - Health checks

3. **Scaling**
   - Async processing
   - Resource management
   - Load balancing
   - Cache management

## Deployment Details

### Server Configuration
- ASGI server (uvicorn)
  - Worker processes based on CPU cores
  - Thread pools for blocking operations
  - Configurable resource limits
  - Health check endpoints

### Container Support
- Docker configuration with multi-stage builds
- Docker Compose for development
- Volume management for persistent data
- Network setup with internal routing

### CORS Middleware
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Scaling Strategy
1. **Horizontal Scaling**
   - Load balancing with nginx
   - Service discovery via consul
   - Sticky sessions for WebSocket
   - Distributed caching

2. **Vertical Scaling**
   - Resource allocation limits
   - Performance tuning parameters
   - Memory optimization settings
   - CPU affinity configuration

## Future Improvements

1. **Plugin System**
   - Hot reload capability
   - Version management
   - Dependency resolution
   - Plugin marketplace

2. **Performance**
   - Caching layer
   - Batch processing
   - Resource optimization
   - Parallel processing

3. **Monitoring**
   - Metrics collection
   - Performance tracking
   - Resource usage
   - System health
