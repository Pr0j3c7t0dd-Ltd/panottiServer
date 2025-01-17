I want to develop a new 'cleanup_files' plugin for the server that will clean up the files in the user-specified directory that relate to the current event processing.

This plugin will have a configuration as follows:
- Directories to include in the clean up (defaults to the 'data' directory in the root of the app).  paths can be relative to the root of the application or absolute paths
- Directories to exclude (defaults to ''). These are the sub-directories that will be excluded from the ones we are including.
# File format to clean up for the event

Here is an example list of files that could be created for an example event ( recording_id - '20250117225532_4AFEA8B6' ):
```
data
├── cleaned_audio
│   └── 20250117225532_4AFEA8B6_mic_bleed_removed_freq.wav
├── meeting_notes_local
│   └── 20250117225532_4AFEA8B6_meeting_notes.md
├── meeting_notes_remote
│   └── 20250117225532_4AFEA8B6_meeting_notes.md
└── transcripts_local
    ├── 20250117225532_4AFEA8B6_merged.md
    ├── 20250117225532_4AFEA8B6_mic_bleed_removed_freq.md
    └── 20250117225532_4AFEA8B6_system_audio.md
```

as you can see, the recording_id prepend is the same for all the files.   We will use this prepend to search for all the files we want to delete for the given event id, while still respecting the include and exclude directory configurations.
# Event types

I want this plugin to subscribe and be triggered by the `desktop_notification.completed` event

I want this new plugin to emit a `cleanup_files.completed` event with the expected payload, per our existing plugins

# Plugin design

This plugin should follow the exact same design pattern already established with the other plugins.  You should do a detail analysis of the other plugins in the app/plugins directory before writing this plugin.

This plugin should contain the following files:
- `__init__.py`
- `.gitignore` (containing only 'plugin.yaml')
- `plugin.py` - the main plugin code
- `plugin.yaml` - the plugin's configuration file
- `plgin.yaml.example` - the example configuration file
- `README.md` - the detailed readme file for the plugin

