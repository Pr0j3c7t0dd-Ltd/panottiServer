# Meeting Notes Remote Plugin

This plugin generates meeting notes from transcripts using the Ollama LLM API. It listens for transcription completion events and automatically generates structured meeting notes.

## Features

- Automatic meeting notes generation from transcripts
- Structured output with sections for:
  - Meeting title and information
  - Executive summary
  - Key discussion points
  - Action items
  - Decisions made
  - Next steps
- Event-driven architecture
- Configurable LLM model and parameters

## Configuration

Copy `plugin.yaml.example` to `plugin.yaml` and adjust the following settings:

```yaml
name: "meeting_notes_remote"
version: "1.0.0"
enabled: true
dependencies: ["audio_transcription"]
config:
  output_directory: "data/meeting_notes_remote"  # Directory for storing generated notes
  ollama_url: "http://localhost:11434/api/generate"  # Ollama API endpoint
  model_name: "llama3.1:latest"  # LLM model to use
  num_ctx: 131072  # Context window size
  max_concurrent_tasks: 4  # Maximum concurrent note generation tasks
```

## Dependencies

- Ollama LLM server running locally or remotely
- Audio transcription plugin

## Usage

The plugin automatically processes transcripts when they become available. No manual intervention is required.

## Output

Meeting notes are generated in markdown format with the following structure:

```markdown
# Meeting Title

## Meeting Information
- Date: [Meeting date and time]
- Duration: [Meeting duration]
- Attendees: [List of attendees]

## Executive Summary
[Brief overview of meeting]

## Key Discussion Points
[Main topics and details]

## Action Items
[List of action items with owners]

## Decisions Made
[List of decisions]

## Next Steps
[Future actions and plans]
```

Files are saved in the configured output directory with the format: `[recording_id]_notes.md` 