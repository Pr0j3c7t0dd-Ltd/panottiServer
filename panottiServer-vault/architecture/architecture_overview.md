# PanottiServer Architecture Overview

## Core Components

### Application Structure
```
app/
├── core/                     # Core system interfaces and protocols
├── models/                   # Domain models and database functionality
├── plugins/                  # Plugin implementations and management
│   ├── base.py              # Base plugin classes and interfaces
│   ├── manager.py           # Plugin lifecycle management
│   ├── events/              # Event system implementation
│   └── [plugin_name]/       # Individual plugin directories
├── utils/                   # Utility functions and configurations
└── tests/                   # Test directory
```

### Key Components

1. **Core System**
   - Defines core interfaces and protocols
   - Implements dependency inversion principle
   - Houses fundamental system contracts

2. **Plugin System**
   - Dynamic plugin discovery and loading
   - Event-driven architecture
   - Plugin configuration via YAML files
   - Structured logging integration

3. **Models**
   - Domain models for business entities
   - Database interactions and schema definitions
   - Event models for system communication

4. **Utils**
   - Logging configuration and management
   - Common utility functions
   - System-wide helpers

## Technology Stack

- **Framework**: FastAPI
- **Dependency Management**: Poetry
- **Testing**: pytest
- **Logging**: structlog
- **Configuration**: Environment variables (.env)
- **Documentation**: Swagger/OpenAPI

## Best Practices

1. **Code Organization**
   - Modular architecture with clear separation of concerns
   - Plugin-based extensibility
   - Type hints throughout the codebase
   - PEP 8 compliance

2. **Development Workflow**
   - Pre-commit hooks for code quality
   - Automated testing
   - Comprehensive logging
   - API documentation

3. **Security**
   - API key authentication
   - HTTPS support
   - Environment-based configuration
   - Secure dependency management

## Configuration
- Environment variables loaded via `dotenv`
- Centralized configuration management
- Separate dev/prod configurations

## Event System
- Event-driven architecture
- Plugin manager for dynamic loading
- Event bus for inter-component communication
- Persistent event store

## Deployment
- ASGI server (`uvicorn`) based deployment
- Docker support
- Environment-specific configurations
- CORS and middleware setup

## Recommended Improvements
1. Reorganize into `src` directory structure
2. Enhance test organization
3. Add dependency auditing
4. Implement structured logging
5. Expand plugin documentation
