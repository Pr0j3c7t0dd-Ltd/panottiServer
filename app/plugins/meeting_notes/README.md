# Meeting Notes Plugin

This plugin automatically generates meeting notes from transcription files using a local Ollama LLM.

## Features

- Listens for 'transcription.completed' events
- Processes transcript files using Ollama LLM
- Generates structured meeting notes in markdown format
- Supports concurrent processing using multi-threading
- Maintains state in SQLite database

## Configuration

The plugin is configured via `plugin.yaml`:

- `output_directory`: Directory where meeting notes will be stored
- `ollama_url`: URL for Ollama API (default: http://localhost:11434/api/generate)
- `model_name`: Ollama model to use (default: llama2:latest)
- `num_ctx`: Context window size (default: 128000)
- `max_concurrent_tasks`: Maximum number of concurrent processing tasks

## Requirements

- Ollama server running locally with the specified model
- Python packages (see requirements.txt):
  - requests>=2.31.0

## Events

### Subscribed Events
- `transcription.completed`: Triggered when a transcript is ready for processing

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
