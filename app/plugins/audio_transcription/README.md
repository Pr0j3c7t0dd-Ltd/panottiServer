# Audio Transcription Plugin

This plugin provides audio transcription functionality using OpenAI's Whisper model through the `faster-whisper` library. It processes audio files and generates timestamped transcripts in markdown format.

## Features
- Transcribes WAV audio files with timestamps
- Supports concurrent processing of multiple files
- Merges multiple transcripts based on relative timestamps
- Integrates with the event system to process files after noise reduction

## Requirements

### System Dependencies
```bash
brew install openai-whisper
```

### Python Dependencies
Add these to your requirements.txt:
```
faster-whisper==0.10.0
```

## Configuration
The plugin can be configured through the `plugin.yaml` file:

```yaml
output_directory: Path to store transcription files
model_name: Whisper model to use (default: "base.en")
max_concurrent_tasks: Maximum number of concurrent transcription tasks
```

## Events

### Listens for:
- `noise_reduction.completed`: Triggered when noise reduction is complete

### Emits:
- `transcription.completed`: When transcription is complete, includes:
  - Original event data
  - Paths to individual transcripts
  - Path to merged transcript file

## Database
The plugin maintains a SQLite table `transcription_tasks` to track processing state:
- recording_id: Identifier from events table
- status: Current processing status
- input_paths: JSON array of input file paths
- output_paths: JSON array of output transcript paths
- merged_output_path: Path to merged transcript
- error_message: Any error details
- timestamps: Created/updated times
