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

## Model Setup

The plugin uses OpenAI's Whisper model through the `faster-whisper` library in offline mode. Before using the plugin, you need to download the model files locally.

### Downloading Models

Use the provided script to download the Whisper model:

```bash
# Download the default model (base.en)
./scripts/download_models.py

# Download a specific model (reccomended)
/app/plugins/audio_transcription/scripts/download_models.py --model base.en

# Available models:
# - tiny.en: Smallest and fastest, less accurate
# - base.en: Good balance of speed and accuracy (default)
# - small.en: Better accuracy, slower than base
# - medium.en: High accuracy, slower
# - large-v2: Best accuracy, much slower
```

Models will be downloaded to `/models/whisper` in the project root by default. You can specify a different location using the `--output-dir` option:

```bash
./scripts/download_models.py --model base.en --output-dir /path/to/models
```

## Configuration
The plugin can be configured through the `plugin.yaml` file:

```yaml
output_directory: Path to store transcription files
model_name: Whisper model to use (default: "base.en")
max_concurrent_tasks: Maximum number of concurrent transcription tasks
```

Alternatively, you can configure the plugin through environment variables or the plugin configuration:

```yaml
audio_transcription:
  model_name: "base.en"  # Which model to use
  model_dir: "/path/to/models/whisper"  # Optional: Override default model directory
  max_concurrent_tasks: 4  # Number of concurrent transcription tasks
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
