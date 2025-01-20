# Meeting Notes Plugin

This plugin automatically generates meeting notes from transcription files using a local Ollama LLM.

## Features

- Listens for 'transcription_local.completed' events
- Processes transcript files using Ollama LLM
- Generates structured meeting notes in markdown format
- Supports concurrent processing using multi-threading
- Maintains state in SQLite database

## Configuration

To set up the plugin configuration:

1. Copy `plugin.yaml.example` to `plugin.yaml`
2. Update the configuration values according to your needs

The configuration supports the following options:

- `output_directory`: Directory where meeting notes will be stored
- `ollama_url`: URL for Ollama API. Options:
  - Local Ollama (default): `http://localhost:11434/api/generate`
  - Docker on macOS/Windows using host Ollama: `http://host.docker.internal:11434/api/generate`
  - Docker on Linux using host Ollama: `http://172.17.0.1:11434/api/generate`
- `model_name`: Ollama model to use (default: llama2:latest)
- `num_ctx`: Context window size (default: 128000)
- `max_concurrent_tasks`: Maximum number of concurrent processing tasks
- `timeout`: Request timeout in seconds (default: 900)

## Requirements

- Ollama server running either (all options requires setting the appropriate `ollama_url`):
  - On the host machine, running the app as a standalone process (default)
  - On the host machine, but called from the app running in Docker
  - Runnig ollama within Docker (not recommended)
- Python packages (see requirements.txt):
  - requests>=2.31.0

## Events

### Subscribed Events
- `transcription_local.completed`: Triggered when a transcript is ready for processing

### Emitted Events
- `meeting_notes.completed`: Emitted when meeting notes generation is complete
  - Includes original event data plus path to generated meeting notes

## Database

The plugin maintains a table `meeting_notes_tasks` in SQLite with the following schema:
- `id`: Primary key
- `recording_id`: ID from the events table
- `input_path`: Path to input transcript file
- `output_path`: Path to generated meeting notes
- `status`: Processing status
- `created_at`: Creation timestamp
- `completed_at`: Completion timestamp
