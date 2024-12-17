# Noise Reduction Plugin

This plugin processes audio files to remove background noise from microphone recordings by using the system audio as a noise profile.

## Features
- Listens for 'recording_ended' events
- Processes microphone audio using system audio as noise profile
- Multi-threaded processing for concurrent operations
- Stores processing state in SQLite database
- Emits completion events with cleaned file information

## Dependencies
Add these to your requirements.txt:
```
numpy>=1.24.0
scipy>=1.10.0
```

## Configuration
Plugin configuration in plugin.yaml:
```yaml
name: noise_reduction
enabled: true
config:
  output_directory: /path/to/output/directory  # Directory to store cleaned audio files
  noise_reduce_factor: 0.7  # Amount of noise reduction (0 to 1)
  max_concurrent_tasks: 4  # Maximum number of concurrent processing tasks
```

## Database Schema
The plugin creates a table to track processing state:
```sql
CREATE TABLE IF NOT EXISTS noise_reduction_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'pending', 'processing', 'completed', 'failed'
    input_mic_path TEXT,
    input_sys_path TEXT,
    output_path TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Events
### Listens for:
- `recording_ended`: Triggered when a recording session ends

### Emits:
- `noise_reduction.completed`: When processing is complete
  ```python
  {
      "recording_id": str,
      "original_event": dict,  # Original recording_ended event data
      "output_file": str,  # Path to cleaned audio file
      "status": str  # "success" or "skipped"
  }
  ```
