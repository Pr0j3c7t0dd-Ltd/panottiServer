# Desktop Notifier Plugin

> ⚠️ **Note**: This plugin is not compatible with Docker deployments as it requires macOS-specific `terminal-notifier` for desktop notifications. If you're running the server in Docker, this plugin should be disabled in your configuration.

This plugin provides desktop notifications when meeting notes are completed. It listens for the 'meeting_notes.completed' event and sends notifications using terminal-notifier.

## Requirements

### System Dependencies
- terminal-notifier (macOS only)
  ```bash
  brew install terminal-notifier
  ```

### Python Dependencies
No additional Python packages required beyond the core application dependencies.

## Configuration

To configure the plugin:

1. Copy `plugin.yaml.example` to `plugin.yaml`
2. Update the configuration values according to your needs

The plugin configuration supports the following options:

- `auto_open_notes`: Boolean (default: false) - If true, automatically opens the meeting notes file when notification is shown
- `max_concurrent_tasks`: Integer (default: 4) - Maximum number of concurrent notification tasks

## Events

### Subscribed Events
- `meeting_notes.completed` - Triggered when meeting notes generation is complete

### Published Events
- `desktop_notification.completed` - Emitted after notification is shown
  - Payload includes all information from the original meeting_notes.completed event

## Usage

The plugin will automatically send desktop notifications when meeting notes are completed. If `auto_open_notes` is enabled, it will also open the notes file automatically.
