# Meeting Notes Remote Plugin

This plugin generates meeting notes from transcripts using various LLM providers (OpenAI, Anthropic, or Google). It listens for transcription completion events and automatically generates structured meeting notes.

## Features

- Automatic meeting notes generation from transcripts
- Support for multiple LLM providers:
  - OpenAI (GPT-4)
  - Anthropic (Claude 3 Sonnet)
  - Google (Gemini 1.5 Pro)
- Structured output with sections for:
  - Meeting title and information
  - Executive summary
  - Key discussion points
  - Action items
  - Decisions made
  - Next steps
- Event-driven architecture
- Configurable provider and model parameters

## Configuration

Copy `plugin.yaml.example` to `plugin.yaml` and adjust the following settings:

```yaml
name: "meeting_notes_remote"
version: "1.0.0"
enabled: true
dependencies: ["audio_transcription"]
config:
  output_directory: "data/meeting_notes_remote"  # Directory for storing generated notes
  provider: "openai"  # Supported values: openai, anthropic, google
  max_concurrent_tasks: 4  # Maximum concurrent note generation tasks
  timeout: 600  # Request timeout in seconds
  openai:
    api_key: "your-openai-api-key"
    model: "gpt-4"
  anthropic:
    api_key: "your-anthropic-api-key"
    model: "claude-3-sonnet-20240229"
  google:
    api_key: "your-google-api-key"
    model: "gemini-1.5-pro"
```

## Dependencies

- One of the following API keys:
  - OpenAI API key
  - Anthropic API key
  - Google API key
- Audio transcription plugin
- Python packages (installed via requirements.txt):
  - openai>=1.12.0
  - anthropic>=0.18.1
  - google-generativeai>=0.3.2

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