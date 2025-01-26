# Plugin System Architecture

## Overview

The plugin system in PanottiServer is designed for extensibility and modularity. It uses a combination of YAML configuration and Python implementation files to define and manage plugins. The system supports dynamic loading, event-driven communication, and robust error handling.

## Event System Migration Complete

The event system has been fully migrated from `app/plugins/events/` to `app.core.events`. All event-related functionality is now centralized in the core package:

```python
# Use the core event system implementation
from app.core.events import (
    EventBus,
    Event,
    EventContext,
    EventPriority,
    EventHandler
)
```

The new implementation in `app.core.events` provides:
- Robust async event processing
- Event persistence and replay capabilities
- Type-safe event models
- Dedicated event handlers
- Improved error handling and logging

## Plugin Structure

Each plugin follows this structure:
```
plugins/
└── [plugin_name]/
    ├── plugin.yaml         # Plugin configuration (from plugin.yaml.example)
    ├── plugin.yaml.example # Example configuration template
    ├── plugin.py          # Plugin implementation
    ├── __init__.py        # Package initialization
    ├── .gitignore         # Git ignore file (ignores plugin.yaml)
    └── README.md          # Plugin documentation
```

## Plugin Configuration

Plugins are configured using `plugin.yaml` files:

```yaml
name: plugin_name          # Unique identifier for the plugin
version: "1.0.0"          # Semantic versioning
enabled: true             # Whether the plugin is active
dependencies: []          # List of required plugins
config:                   # Plugin-specific configuration
    max_concurrent_tasks: 4  # Common configuration for concurrent processing
    output_directory: "data/[plugin_name]"  # Plugin output location
    # ... other plugin-specific settings
```

## Core Components

### PluginBase Class
- Abstract base class for all plugins
- Provides lifecycle management (initialize, shutdown)
- Handles event bus integration
- Implements logging and error handling
- Tracks initialization state

### PluginManager
- Handles plugin discovery and loading
- Manages plugin configurations
- Validates dependencies
- Provides plugin lifecycle management
- Implements robust error handling and logging

### Event System Integration
- Event-driven architecture using core EventBus (`app.core.events`)
- Supports asynchronous event handling with robust error handling
- Provides structured event context and priority levels
- Enables plugin-to-plugin communication through centralized event bus
- Implements event persistence and cleanup
- Uses Pydantic models for type safety and validation
- Supports event correlation and tracing
- Provides comprehensive logging and monitoring

The event system is implemented in `app.core.events` and provides:
- `EventBus`: Central message broker with async support
- `Event`: Base event model with validation
- `EventContext`: Structured context for event metadata
- `EventPriority`: Priority levels (LOW, NORMAL, HIGH, CRITICAL)

Example usage:
```python
from app.core.events import Event, EventContext, EventPriority

# Create and emit an event
event = Event.create(
    name="recording_started",
    data={"recording_id": "rec_123"},
    correlation_id="corr_id",
    source_plugin="my_plugin",
    priority=EventPriority.HIGH
)
await self.event_bus.publish(event)
```

## Plugin Development Guide

### Creating New Plugins

1. Create plugin directory structure:
   ```bash
   mkdir -p app/plugins/[plugin_name]
   ```

2. Create required files:
   - `plugin.yaml.example` with configuration template
   - `plugin.py` implementing `PluginBase`
   - `__init__.py` exposing the plugin class
   - `.gitignore` to exclude `plugin.yaml`
   - `README.md` with plugin documentation

3. Implement the plugin class:
   ```python
   from app.core.plugins import PluginBase, PluginConfig
   
   class MyPlugin(PluginBase):
       def __init__(self, config: PluginConfig, event_bus=None):
           super().__init__(config, event_bus)
           # Plugin-specific initialization
   ```

### Best Practices

1. **Configuration Management**
   - Use `plugin.yaml.example` as a template
   - Document all configuration options
   - Implement validation for config values
   - Use type hints for configuration

2. **Error Handling**
   - Implement comprehensive error handling
   - Use structured logging with context
   - Gracefully handle initialization failures
   - Properly clean up resources on shutdown

3. **Event Handling**
   - Use typed event models
   - Implement proper error handling in event handlers
   - Use async/await for event processing
   - Handle event context properly

4. **Resource Management**
   - Use thread pools for CPU-intensive tasks
   - Implement proper cleanup in shutdown
   - Handle concurrent processing limits
   - Use async IO for I/O-bound operations

5. **Testing**
   - Write unit tests for plugin functionality
   - Test event handling
   - Mock dependencies and external services
   - Test configuration validation

6. **Documentation**
   - Maintain comprehensive README
   - Document configuration options
   - Include usage examples
   - Document dependencies and requirements

## Plugin Examples

The codebase includes several reference implementations:

1. **Example Plugin**
   - Reference implementation demonstrating best practices
   - Complete documentation and type hints
   - Example event handling and configuration

2. **Noise Reduction Plugin**
   - Audio processing functionality
   - Complex configuration options
   - Resource management example

3. **Meeting Notes Plugins**
   - Both local and remote implementations
   - Dependency management example
   - Integration with external services

## Security Considerations

1. **Configuration Security**
   - Sensitive configuration in `plugin.yaml`
   - Example configurations in version control
   - Proper gitignore setup

2. **Resource Access**
   - Controlled access to system resources
   - Proper permission handling
   - Secure external service integration

3. **Error Handling**
   - No sensitive data in error logs
   - Proper exception handling
   - Secure cleanup on failures
