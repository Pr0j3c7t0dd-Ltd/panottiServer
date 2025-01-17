# Cleanup Files Plugin

This plugin automatically cleans up files related to a recording event after processing is complete.

## Overview

The Cleanup Files plugin monitors for completed desktop notifications and removes associated files from configured directories. It helps maintain a clean workspace by removing processed files that are no longer needed.

## Configuration

The plugin is configured through `plugin.yaml`:

- **name**: cleanup_files
- **version**: Plugin version following semantic versioning
- **enabled**: Whether the plugin is active
- **config**:
  - **include_dirs**: List of directories to clean (defaults to ["data"])
  - **exclude_dirs**: List of subdirectories to exclude from cleanup (defaults to [])
  - **cleanup_delay**: Number of seconds to wait before starting cleanup (defaults to 0)

### Path Configuration

Both `include_dirs` and `exclude_dirs` support relative and absolute paths:

#### Relative Paths
Relative paths are resolved from the application root directory:

```yaml
config:
  include_dirs: ["data", "data/transcripts_local"]  # Cleans files in {app_root}/data and transcripts_local
  exclude_dirs: ["data/meeting_notes_local"]  # Excludes the meeting_notes_local subdirectory
  cleanup_delay: 5  # Wait 5 seconds before starting cleanup
```

#### Absolute Paths
Absolute paths must start with a forward slash:

```yaml
config:
  include_dirs: ["/Users/username/recordings", "/var/log/panotti"]  # Cleans files in specific directories
  exclude_dirs: ["/Users/username/recordings/archived"]  # Excludes archived recordings
  cleanup_delay: 10  # Wait 10 seconds before starting cleanup
```

#### Mixed Configuration Example
You can mix relative and absolute paths:

```yaml
config:
  include_dirs: ["data", "data/transcripts_local", "/Users/username/recordings"]  # Mix of relative and absolute paths
  exclude_dirs: ["data/meeting_notes_local", "/Users/username/recordings/archived"]  # Mix of relative and absolute exclusions
  cleanup_delay: 3  # Wait 3 seconds before starting cleanup
```

### Security Note
When using absolute paths:
- Ensure the application has proper permissions to access and modify the directories
- Be cautious with system directories to prevent accidental deletion
- Consider using relative paths when possible for better portability

## Installation

1. Copy `plugin.yaml.example` to `plugin.yaml`
2. Modify configuration as needed
3. Restart the server

## Events

### Subscribes to:
- `desktop_notification.completed`: Triggered when desktop notification is complete

### Emits:
- `cleanup_files.completed`: Emitted after files are cleaned up
  - Payload includes:
    - recording_id: ID of the processed recording
    - cleaned_files: List of files that were cleaned up
    - status: Completion status