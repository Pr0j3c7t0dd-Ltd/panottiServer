# Architecture Overview

## Framework and Structure
- **FastAPI** is used as the web framework.
- Organized into modules such as `models`, `plugins`, and `utils` for a clean and modular structure.

## Key Components
- **Models**: Handles data structures and database interactions.
- **Plugins**: Manages additional functionalities and event handling.
- **Utils**: Contains utility functions like logging configuration.
- **Tests**: Includes test files to ensure code quality and functionality.

## Configuration
- Environment variables loaded using `dotenv`.
- Logging set up through a custom configuration.

## Event System
- Event bus and store initialized for managing recording events.
- Plugin manager handles plugins dynamically.

## Security
- API Key security implemented to protect endpoints.

## Middleware
- CORS middleware configured for cross-origin requests.

## Deployment
- Designed to run with `uvicorn`, a fast ASGI server.
